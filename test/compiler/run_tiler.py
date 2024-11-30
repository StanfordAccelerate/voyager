import argparse
import interstellar

from google.protobuf import text_format

from quantized_training.codegen import param_pb2
from proto import tiling_pb2


def calculate_cost(architecture, layer, mapping):
    """
    Calculate the cost of a given mapping. Print the schedule and loop nest, as well as the number of accesses per level and total energy.
    """
    print("*" * 80)
    print("Details of the mapping:")
    interstellar.utils.print_tiling(mapping)

    total_cost, total_access_cost, access_list = interstellar.cost_model.get_cost(
        architecture, mapping, layer
    )

    valid_mapping = valid = [
        interstellar.cost_model.valid_blocking_size_current_level(
            architecture, mapping, layer, i
        )
        for i in range(3)
    ]

    print("Valid mapping:", valid_mapping)
    # print total_cost in scientific notation
    print("Total energy (pJ): {:.2e}".format(total_cost))
    print("Total access energy (pJ):", total_access_cost)
    print("Access breakdown:")
    print(access_list)

    print("*" * 80)


def find_best_schedule():
    # Create an accelerator configuration
    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],  # PE register file
            [1024 * 16, 1024 * 16, 1024 * 16],  # L1 buffer
            [12 * 1024 * 1024],  # L2 main memory
        ],
        memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0]],
        buf_access_cost_list=[[1, 1, 1], [10, 10, 10], [100]],
        buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0]],
        para_count_list=[256, 1, 1],  # 16x16
        mac_capacity=0,
        partition_mode=[0, 0, 0],
        invalid_underutilized=False,
    )

    # submodule_1
    layer = interstellar.Layer(
        nifm=64,
        nofm=64,
        wofm=56,
        hofm=56,
        wfil=3,
        hfil=3,
    )

    # create schedule
    CK_dataflow = {
        "schedule_hint": {
            "IC": {
                "level0": {"order": 1, "partitioning_size": 16},
                "level1": {"order": -1},
                "level2": {"order": 0},
            },
            "OC": {"level0": {"order": 0, "partitioning_size": 16}},
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

    # C|K dataflow
    schedule = interstellar.extract_input.extract_schedule_info(CK_dataflow, 3)
    schedule = interstellar.Schedule(
        schedule["schedule_hint"], schedule["partition_loops"]
    )

    cost, mapping, perf = interstellar.optimizer.opt_optimizer(
        architecture, layer, schedule, True
    )

    calculate_cost(architecture, layer, mapping)
    # interstellar.utils.print_best_schedule(mapping)
    # interstellar.utils.print_loop_nest(mapping)
    # interstellar.cost_model.get_fl_access(0, mapping, layer)

    custom_mapping = interstellar.MappingPoint(
        loop_order_list=[
            (6, 2, 6),
            (6, 3, 6),
            (6, 0, 1),
            (6, 1, 2),
            (0, 6, 0),
            (1, 4, 6),
            (6, 6, 6),
        ],
        loop_blockings_list=[
            (1, 3, 1),
            (1, 3, 1),
            (1, 28, 2),
            (1, 28, 2),
            (1, 1, 4),
            (1, 4, 1),
            (1, 1, 1),
        ],
        loop_partitionings_list=[
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (16, 1, 1),
            (16, 1, 1),
            (1, 1, 1),
        ],
        para_loop_dim_list=([[4], [5]], None, None),
    )
    # calculate_cost(architecture, layer, custom_mapping)

    # hand-mapped schedule
    custom_mapping = interstellar.MappingPoint(
        loop_order_list=[
            (6, 2, 6),
            (6, 3, 6),
            (6, 0, 2),
            (6, 1, 3),
            (0, 6, 1),
            (1, 4, 0),
            (6, 6, 6),
        ],
        loop_blockings_list=[
            (1, 3, 1),
            (1, 3, 1),
            (1, 28, 2),
            (1, 28, 2),
            (1, 1, 4),
            (1, 1, 4),
            (1, 1, 1),
        ],
        loop_partitionings_list=[
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (16, 1, 1),
            (16, 1, 1),
            (1, 1, 1),
        ],
        para_loop_dim_list=([[4], [5]], None, None),
    )
    calculate_cost(architecture, layer, custom_mapping)


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
    args = parser.parse_args()

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

    # read the params.pb file
    params = param_pb2.ModelParams()
    with open(f"{args.codegen_dir}/params.pb", "rb") as f:
        params.ParseFromString(f.read())

    # generate tilings for all matrix operations
    tilings = tiling_pb2.ModelTiling()
    for param in params.params:
        if param.HasField("matrix_param"):
            input_shape = param.matrix_param.input.shape
            if len(input_shape) == 4:
                input_channels = input_shape[1]
                height = input_shape[2]
                width = input_shape[3]
            else:
                print(f"Unsupported input shape: {input_shape}, skipping")
                continue

            weights_shape = param.matrix_param.weight.shape
            if len(weights_shape) == 4:
                output_channels = weights_shape[0]
                kernel_height = weights_shape[2]
                kernel_width = weights_shape[3]
            else:
                print(f"Unsupported weights shape: {weights_shape}, skipping")
                continue

            if input_channels == 3:
                print(f"Skipping input with {input_channels} channels")
                continue

            stride = param.matrix_param.stride[0]

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

            cost, mapping, perf = interstellar.optimizer.opt_optimizer(
                architecture, layer, schedule, True
            )

            interstellar.utils.print_tiling(mapping)

            tiling = tiling_pb2.Tiling()
            tiling.name = param.name
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
            tilings.tilings.append(tiling)

        # write the tilings to a file
        with open(f"{args.codegen_dir}/tilings.txtpb", "w") as f:
            f.write(text_format.MessageToString(tilings))


if __name__ == "__main__":
    main()
