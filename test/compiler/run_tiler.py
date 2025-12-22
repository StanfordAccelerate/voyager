import argparse
import interstellar
import logging
import math
import os
import re
from deepdiff import DeepDiff
from google.protobuf import text_format
from google.protobuf.json_format import MessageToDict
from voyager_compiler.codegen import param_pb2
from proto import tiling_pb2


logger = logging.getLogger(__name__)


def get_dtype_width(dtype: str):
    bit_search = re.search(r"[^\d](\d+)(_.*)?$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    return int(bit_search.groups()[0])


class RuntimeCalculator:
    def __init__(self, operation, double_buffered_accum_buffer):
        self.operation = operation
        self.double_buffered_accum_buffer = double_buffered_accum_buffer

    def calculate_runtime(self, architecture, layer, mapping):
        # assume IC is unrolled vertically
        # does not handle replication
        sa_weight_loading_time = mapping.loop_partitionings[interstellar.le.IC][0] + 2

        sa_latency = (
            mapping.loop_partitionings[interstellar.le.IC][0] * 3
            + mapping.loop_partitionings[interstellar.le.OC][0]
        )

        # index of first loop that isn't OX or OY
        first_non_ox_oy_index = 6
        for i in range(interstellar.le.NUM):
            if i == interstellar.le.OX or i == interstellar.le.OY:
                continue
            if mapping.loop_orders[i][1] < first_non_ox_oy_index:
                first_non_ox_oy_index = mapping.loop_orders[i][1]

        # calculate how big a weight reuse tile is
        # weights are reused in the OX and OY loops, so a weight reuse tile is product of
        # all loops from the first loop to the first non-(OX or OY) loop
        weight_reuse_tile_size = 1
        for i in range(interstellar.le.NUM):
            if mapping.loop_orders[i][1] < first_non_ox_oy_index:
                weight_reuse_tile_size *= mapping.loop_blockings[i][1]

        # calculate how long it takes to compute a weight_reuse_tile
        # this is the max of time to load the weights into the systolic array and weight reuse tile size
        weight_reuse_tile_time = max(sa_weight_loading_time, weight_reuse_tile_size)

        # calculate number of remaining L1 tiles
        num_remaining_l1_tiles = 1
        for i in range(interstellar.le.NUM):
            if mapping.loop_orders[i][1] >= first_non_ox_oy_index:
                num_remaining_l1_tiles *= mapping.loop_blockings[i][1]
        # include the reduction loop at the L2 level
        num_remaining_l1_tiles *= mapping.loop_blockings[interstellar.le.IC][2]

        # calculate time for computation at the L1 level
        computation_l1_time = weight_reuse_tile_time * num_remaining_l1_tiles

        input_relevant_loops = [
            interstellar.le.IC,
            interstellar.le.OY,
            interstellar.le.OX,
        ]
        input_buffer_loading_size = 1
        for loop in input_relevant_loops:
            input_buffer_loading_size *= mapping.loop_blockings[loop][1]
        # currently assume that each value in the input buffer is loaded in one cycle
        input_buffer_loading_time = input_buffer_loading_size

        if self.operation.WhichOneof("op_type") == "op":
            matrix_op = self.operation.op
        else:
            matrix_op = self.operation.fused_op.op_list[0]
        input_dtype = matrix_op.kwargs["input"].tensor.dtype
        input_dtype_width = get_dtype_width(input_dtype)

        if self.operation.HasField("output"):
            output_dtype = self.operation.output.dtype
        else:
            output_dtype = self.operation.outputs.tensors[1].dtype
        output_dtype_width = get_dtype_width(output_dtype)

        weight_relevant_loops = [
            interstellar.le.IC,
            interstellar.le.OC,
            interstellar.le.FY,
            interstellar.le.FX,
        ]
        weight_buffer_loading_size = 1
        for loop in weight_relevant_loops:
            weight_buffer_loading_size *= mapping.loop_blockings[loop][1]
        # include the unrolled reduction loop
        weight_buffer_loading_size *= mapping.loop_partitionings[interstellar.le.IC][0]
        # currently assume that each value in the weight buffer is loaded in one cycle
        weight_buffer_loading_time = weight_buffer_loading_size

        # calculate time for vector unit to process values from the accumulation buffer
        # for now, let's assume that all vector unit operations flow through the vector unit pipeline once.
        # the only thing that matters then is if the other operand or the outputs are in high precision, which will 2x the time
        output_relevant_loops = [
            interstellar.le.OC,
            interstellar.le.OY,
            interstellar.le.OX,
        ]
        output_size = 1
        for loop in output_relevant_loops:
            output_size *= mapping.loop_blockings[loop][1]
        vector_unit_time = output_size

        requires_high_precision = False
        if output_dtype_width > input_dtype_width:
            requires_high_precision = True
        elif self.operation.WhichOneof("op_type") == "fused_op":
            # check if any of the operands in fused ops are in high precision
            for vector_op in self.operation.fused_op.op_list[1:]:
                for arg in vector_op.kwargs.values():
                    if not arg.HasField("tensor") or not arg.tensor.HasField("memory"):
                        continue

                    dtype_width = get_dtype_width(arg.tensor.dtype)
                    if dtype_width > input_dtype_width:
                        requires_high_precision = True
                        break

        if requires_high_precision:
            vector_unit_time *= 2

        using_double_buffer_accum_buffer = (
            self.double_buffered_accum_buffer and requires_high_precision
        )

        if not using_double_buffer_accum_buffer:
            # if we are not using double buffered accumulation buffer, the vector unit time should not factor in to the computation of l1_time
            l1_time = max(
                computation_l1_time,
                input_buffer_loading_time,
                weight_buffer_loading_time,
            )
        else:
            l1_time = max(
                computation_l1_time,
                input_buffer_loading_time,
                weight_buffer_loading_time,
                vector_unit_time,
            )

        l2_blocks = 1
        for i in range(interstellar.le.NUM):
            # don't count the reduction loop here, since it's counted towards the number of L1 tiles
            if i == interstellar.le.IC:
                continue
            l2_blocks *= mapping.loop_blockings[i][2]

        if self.double_buffered_accum_buffer:
            total_time = (
                # initial buffer loading time
                max(input_buffer_loading_time, weight_buffer_loading_time)
                + l2_blocks * l1_time
                # time for last tile to be processed by vector unit
                + vector_unit_time
            )
        else:
            # if there's no double buffered accumulation buffer, we need to account for any extra time taken by the vector unit
            # this extra time is any stalling that occurs when the vector unit uses high precision
            if requires_high_precision:
                extra_vector_unit_time = output_size
            else:
                extra_vector_unit_time = 0

            total_time = (
                # initial buffer loading time
                max(input_buffer_loading_time, weight_buffer_loading_time)
                + l2_blocks * l1_time
                + extra_vector_unit_time
            )

        return total_time


def delete_nested_keys(data, key):
    if isinstance(data, dict):
        for k in list(data.keys()):
            if k == key:
                del data[k]
            else:
                delete_nested_keys(data[k], key)
    elif isinstance(data, list):
        for item in data:
            delete_nested_keys(item, key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codegen_dir", type=str, required=True, help="Directory for compiler output"
    )
    parser.add_argument(
        "--IC_dimension", type=int, required=True, help="Number of IC dimensions"
    )
    parser.add_argument(
        "--OC_dimension", type=int, required=True, help="Number of OC dimensions"
    )
    parser.add_argument(
        "--input_buffer_size",
        type=int,
        required=True,
        help="Size of input buffer (in # elements)",
    )
    parser.add_argument(
        "--weight_buffer_size",
        type=int,
        required=True,
        help="Size of weight buffer (in # elements)",
    )
    parser.add_argument(
        "--accum_buffer_size",
        type=int,
        required=True,
        help="Size of output buffer (in # elements)",
    )
    parser.add_argument(
        "--double_buffered_accum_buffer",
        action="store_true",
        help="Use double buffered accumulation buffer",
    )
    args = parser.parse_args()

    # Define a cache dictionary
    cache = {}

    # Create an accelerator configuration
    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],  # PE register file
            [
                args.input_buffer_size * args.IC_dimension,
                args.accum_buffer_size * args.OC_dimension,
                args.weight_buffer_size * args.OC_dimension,
            ],  # L1 buffer
            [12 * 1024 * 1024],  # L2 main memory
        ],
        memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0]],
        buf_access_cost_list=[[1, 1, 1], [10, 10, 10], [100]],
        buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0]],
        para_count_list=[args.IC_dimension * args.OC_dimension, 1, 1],
        mac_capacity=0,
        partition_mode=[0, 0, 0],
        invalid_underutilized=False,
    )

    # Restrict dataflow to what we can handle
    schedule_constraint = {
        "schedule_hint": {
            "IC": {
                "level0": {"order": 1, "partitioning_size": args.IC_dimension},
                "level1": {"order": -1},
                "level2": {"order": 0},
            },
            "OC": {"level0": {"order": 0, "partitioning_size": args.OC_dimension}},
            "FX": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
            },
            "FY": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
            },
        }
    }
    schedule = interstellar.extract_input.extract_schedule_info(schedule_constraint, 3)
    schedule = interstellar.Schedule(
        schedule["schedule_hint"], schedule["partition_loops"]
    )

    # read the model.txt file
    with open(f"{args.codegen_dir}/model.txt", "r") as f:
        contents = f.read()
    params = param_pb2.Model()
    text_format.Parse(contents, params)

    unique_layers = {}

    # generate tilings for all matrix operations
    tilings = tiling_pb2.ModelTiling()
    for param in params.ops:
        if param.WhichOneof("op_type") == "op":
            matrix_op = param.op
            param_name = matrix_op.name
        else:
            matrix_op = param.fused_op.op_list[0]
            param_name = param.fused_op.name

        if matrix_op.target not in [
            "conv2d",
            "linear",
            "matmul",
            "conv2d_mx",
            "linear_mx",
            "matmul_mx",
        ]:
            print(f"Unsupported operation: {matrix_op.target}, skipping")
            continue

        def get_shape(tensor):
            return (
                tensor.tiled_shape
                if "SOC_SIM" in os.environ and tensor.tiled_shape
                else tensor.shape
            )

        # FC doesn't need tiling
        input = matrix_op.kwargs["input"].tensor
        input_shape = get_shape(input)
        if math.prod(input_shape[:-1]) == 1:
            print(f"Skipping {param_name} as it is a fully connected layer with no tiling")
            continue

        # Skip first layer of torchvision models
        if matrix_op.target in ["conv2d", "conv2d_mx"] and input_shape[-1] == 3:
            print(f"Skipping {param_name} as it is a 3-channel input")
            continue
        
        if "groups" in matrix_op.kwargs:
            group = matrix_op.kwargs["groups"].int_value
            if group != 1:
                print("Skipping DwC")
                continue

        print(f"Processing {param_name} with target {matrix_op.target}")

        is_matmul = matrix_op.target in ["matmul", "matmul_mx"]
        weight = matrix_op.kwargs["other" if is_matmul else "weight"].tensor
        if weight.HasField("reshape"):
            weights_shape = weight.reshape.kwargs["output_shape"].int_list.values
        else:
            weights_shape = get_shape(weight)

        output = param.output if param.HasField("output") else param.outputs.tensors[1]

        # HACK: for operations with reshape, look at the input shape
        if output.HasField("reshape"):
            output_shape = matrix_op.kwargs["input"].tensor.shape
        else:
            output_shape = get_shape(output)

        if matrix_op.target in ["conv2d", "conv2d_mx"]:
            assert len(weights_shape) == 4, "Expected weights shape to be 4D for convolution"
            kernel_width = weights_shape[0]
            kernel_height = weights_shape[1]
            input_channels = weights_shape[2]
            output_channels = weights_shape[3]
            height = output_shape[1]
            width = output_shape[2]
        else:
            if len(weights_shape) != 2:
                logger.warning(
                    f"Unexpected weights shape {weights_shape} for "
                    f"{matrix_op.target}, skipping"
                )
                continue
            kernel_width = 1
            kernel_height = 1
            input_channels = weights_shape[0]
            output_channels = weights_shape[1]
            height = 1
            width = math.prod(output_shape[:-1])

        if "stride" in matrix_op.kwargs:
            stride = matrix_op.kwargs["stride"].int_list.values[0]
        else:
            stride = 1

        print(f"Input channels: {input_channels}")
        print(f"Output channels: {output_channels}")
        print(f"Output height: {height}")
        print(f"Output width: {width}")
        print(f"Kernel height: {kernel_height}")
        print(f"Kernel width: {kernel_width}")
        print(f"Stride: {stride}")

        layer = interstellar.Layer(
            nifm=input_channels,
            nofm=output_channels,
            wofm=width,
            hofm=height,
            wfil=kernel_width,
            hfil=kernel_height,
            wstd=stride,
            hstd=stride,
        )

        print(f"Finding tiling for {param_name}")

        # remove the name, memory, and node fields from the op
        params_dict = MessageToDict(param, preserving_proto_field_name=True)
        delete_nested_keys(params_dict, "name")
        delete_nested_keys(params_dict, "memory")
        delete_nested_keys(params_dict, "scratchpad")
        delete_nested_keys(params_dict, "node")

        matched_param = None
        for op_name, op_val in unique_layers.items():
            if not DeepDiff(params_dict, op_val, ignore_order=True):
                matched_param = op_name
                break

        # check if the layer is already in the cache
        if matched_param is not None:
            print(f"Layer {param_name} is identical to cached layer {matched_param}")
            cost, runtime, mapping, perf, total_cost, total_access_cost, access_list = (
                cache[matched_param]
            )
        else:
            print(f"Layer {param_name} not in cache, calling Interstellar optimizer")
            unique_layers[param_name] = params_dict

            runtime_calculator = RuntimeCalculator(
                param, args.double_buffered_accum_buffer
            )

            cost, runtime, mapping, perf = interstellar.optimizer.opt_optimizer(
                architecture,
                layer,
                schedule,
                runtime_calculator.calculate_runtime,
                True,
            )

            total_cost, total_access_cost, access_list = (
                interstellar.cost_model.get_cost(architecture, mapping, layer)
            )

            cache[param_name] = (
                cost,
                runtime,
                mapping,
                perf,
                total_cost,
                total_access_cost,
                access_list,
            )

        interstellar.utils.print_tiling(mapping)
        print(f"Runtime: {runtime}")

        tiling = tiling_pb2.Tiling()
        tiling.name = param_name
        for level in range(1, architecture.num_levels):  # skip PE level
            level_tiling = tiling_pb2.LevelTiling()
            loop_index = 0
            while loop_index < interstellar.le.NUM:
                for loop in range(interstellar.le.NUM):
                    if mapping.loop_orders[loop][level] == loop_index:
                        loop_bound = tiling_pb2.LoopBound()
                        loop_bound.loop = loop
                        loop_bound.bound = mapping.loop_blockings[loop][level]
                        loop_index += 1
                        level_tiling.loop_bounds.append(loop_bound)
                        break
                    else:
                        # if we have reached the last loop, we have no more bounds to add
                        if loop == interstellar.le.NUM - 1:
                            loop_index = interstellar.le.NUM

            tiling.level_tilings.append(level_tiling)

            access_count = tiling_pb2.LevelAccessCount()
            access_count.input_access_count = access_list[level][0]
            access_count.output_access_count = access_list[level][1]
            access_count.weight_access_count = access_list[level][2]

            tiling.level_access_counts.append(access_count)

        tilings.tilings.append(tiling)

    # write the tilings to a file
    with open(
        f"{args.codegen_dir}/{args.IC_dimension}x{args.OC_dimension}_{args.input_buffer_size}x{args.weight_buffer_size}x{args.accum_buffer_size}_{str(args.double_buffered_accum_buffer).lower()}/tilings.txtpb",
        "w",
    ) as f:
        f.write(text_format.MessageToString(tilings))


if __name__ == "__main__":
    main()
