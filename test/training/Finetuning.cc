#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

#define NO_SYSC
// clang-format off
#include "src/PositTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/GoldModel.h"
#include "test/common/SimpleMemoryModel.h"
#include "test/common/Utils.h"
#include "test/mobilebert/MobileBERT.h"
#include "test/training/DTYPE.h"
#include "test/training/DatasetIterator.h"

#define MRPC

// Milestone 2- forward pass with checkpoints saved, backward pass with
// activations recomputed from checkpoint
// In addition, we will set the memory offsets ourselves

#include "memory_plan.h"

#define OPERATION(layer, task) #layer, #task

void getMobileBERTParams(std::string layerName, std::string taskName,
                         SimplifiedParams &params, MemoryMap &memoryMap);

void adjustMemoryOffsets(SimplifiedParams &params, int inputOffset,
                         int weightOffset, int outputOffset, int biasOffset,
                         int residualOffset) {
  params.INPUT_OFFSET = inputOffset;
  params.WEIGHT_OFFSET = weightOffset;
  params.OUTPUT_OFFSET = outputOffset;
  params.BIAS_OFFSET = biasOffset;
  params.RESIDUAL_OFFSET = residualOffset;
}

SimpleMemoryModel<DTYPE> *memory;

void runWorkload(SimplifiedParams params, MemoryMap memoryMap) {
#ifdef FP32
  run_fp_gold_model
#else
  run_custom_posit_gold_model
#endif
      (params, memory->sram + params.INPUT_OFFSET,
       (memoryMap.weights ? memory->rram : memory->sram) + params.WEIGHT_OFFSET,
       memory->sram + params.OUTPUT_OFFSET,
       (memoryMap.bias ? memory->rram : memory->sram) + params.BIAS_OFFSET,
       (params.MERGE_LORA_WEIGHT ? memory->rram + params.RESIDUAL_OFFSET
                                 : memory->sram + params.RESIDUAL_OFFSET),
       memory->sram + params.WEIGHT_RESIDUAL_OFFSET);
}

void run_op(const std::string &layerName, const std::string &task,
            int inputOffset, int weightOffset, int outputOffset, int biasOffset,
            int residualOffset, float step = -1) {
  SimplifiedParams params;
  MemoryMap memoryMap;
  getMobileBERTParams(layerName, task, params, memoryMap);
  adjustMemoryOffsets(params, inputOffset, weightOffset, outputOffset,
                      biasOffset, residualOffset);

  if (params.WEIGHT_UPDATE) {
    if (step == -1) {
      std::cout << "ERROR: step not specified for weight update" << std::endl;
      exit(1);
    }
    params.learningRate = START_LR - START_LR * (step / MAX_TRAINING_STEPS);
  }

  runWorkload(params, memoryMap);
}

void dumpToFile(int address, int size, std::string file) {
  // dump bytes output to file
  std::ofstream myfile(file, std::ios::out | std::ios::binary);
  for (int i = 0; i < size; i++) {
    myfile << (uint8_t)memory->sram[address + i].bits;
  }
  myfile.close();
}

// clang-format off
#include "forward_pass.h"
#include "backward_pass.h"
// clang-format on

void load_sample(DatasetIterator &dataset, bool useStepFolder, int step) {
  std::string sampleFolder = dataset.getSample();

  MobileBERT mobilebert("mobilebert", "inference", "");
  std::vector<Workload> workloads = mobilebert.getFullForwardPass();

  // load the model inputs for this sample
  Workload firstWorkload = workloads[0];

  firstWorkload.files.inputs_file =
      useStepFolder
          ? "models/mobilebert/binary_data/tiny_pretrained/step_" +
                std::to_string(step) + "/activations/mobilebert_embeddings"
          : sampleFolder + "/embedding_output";
  firstWorkload.params.INPUT_OFFSET = INPUT;
  memory->loadModelActivations(firstWorkload.params, firstWorkload.files,
                               firstWorkload.memoryMap, true);

  // for loading attention mask, we make a custom workload
  Workload dummyAttentionMaskWorkload =
      mobilebert.getWorkloadsInRange({"attention_self_attention_scores_0"})[0];
  dummyAttentionMaskWorkload.files.inputs_file = "";
  dummyAttentionMaskWorkload.files.weights_file = "";
  dummyAttentionMaskWorkload.files.bias_file =
      useStepFolder
          ? "models/mobilebert/binary_data/tiny_pretrained/step_" +
                std::to_string(step) + "/activations/mobilebert_attention_mask"
          : sampleFolder + "/attention_mask";

  dummyAttentionMaskWorkload.params.ATTENTION_MASK = true;
  dummyAttentionMaskWorkload.params.RESIDUAL = false;
  dummyAttentionMaskWorkload.params.RELU_GRAD = false;
  dummyAttentionMaskWorkload.params.SOFTMAX_GRAD = false;
  dummyAttentionMaskWorkload.params.WEIGHT_SPLITTING = false;
  dummyAttentionMaskWorkload.memoryMap.bias = SRAM;
  dummyAttentionMaskWorkload.params.BIAS_OFFSET = MASK;
  memory->loadModelActivations(dummyAttentionMaskWorkload.params,
                               dummyAttentionMaskWorkload.files,
                               dummyAttentionMaskWorkload.memoryMap, true);

  if (useStepFolder) {
    // open file
    std::ifstream labelFile(
        "models/mobilebert/binary_data/tiny_pretrained/step_" +
            std::to_string(step) + "/activations/mobilebert_labels",
        std::ios::binary);
    double *tmpValuesArray = new double[1024];
    labelFile.read((char *)tmpValuesArray, 1024 * sizeof(double));
    for (int i = 0; i < 2; i++) {
      memory->sram[LABEL + i] = tmpValuesArray[i];
      std::cout << memory->sram[LABEL + i] << std::endl;
    }
    for (int i = 2; i < 16; i++) {
      memory->sram[LABEL + i] = 0;
    }
  } else {
    // write label, using name of folder as label
    int label = sampleFolder[sampleFolder.size() - 1] - '0';
    std::cout << "label: " << label << std::endl;
    for (int i = 0; i < 2; i++) {
      memory->sram[LABEL + i] = i == label ? 1 : 0;
    }
    for (int i = 2; i < 16; i++) {
      memory->sram[LABEL + i] = 0;
    }
    std::cout << "labels: " << std::endl;
    for (int i = 0; i < 16; i++) {
      std::cout << memory->sram[LABEL + i] << std::endl;
    }
  }

  std::cout << "Loaded sample from " << sampleFolder << std::endl;
}

void initialize_model(const std::string &modelPath) {
  int stepNum = 0;

  // Load weights
  MobileBERT mobilebert("mobilebert", "inference", "");
  std::vector<Workload> forwardPass = mobilebert.getFullForwardPass();
  for (Workload &workload : forwardPass) {
    if (workload.loadWeight) {
      workload.files.weights_file.insert(
          0, modelPath + "step_" + std::to_string(stepNum) + "/weights/");
      workload.files.bias_file.insert(
          0, modelPath + "step_" + std::to_string(stepNum) + "/weights/");
      if (workload.params.ATTENTION_MASK) {
        workload.files.bias_file = "";
      }

      if (workload.name == "mobilebert_encoder_layer_20_classifier") {
        workload.params.ACC_T_WEIGHT = true;
      }

      memory->loadModelParams(workload.params, workload.files,
                              workload.memoryMap, true);
    }
  }
  std::cout << "Loaded pretrained model parameters." << std::endl;

  // Load LoRA weights for each encoder layer
  for (int encoderLayer = 0; encoderLayer < 21; encoderLayer++) {
    // load query LoRA weight
    std::string w_q_lora_file =
        "models/mobilebert/binary_data/tiny_pretrained/step_" +
        std::to_string(stepNum) +
        "/weights/"
        "mobilebert_encoder_layer_" +
        std::to_string(encoderLayer) +
        "_attention_self_query_lora_A_default_weight";
    double *w_q_lora_weights =
        readFileAsDouble(w_q_lora_file, LORA_WQ_A_SIZE / 2, true);

    for (int i = 0; i < LORA_WQ_A_SIZE / 2; i++) {
#ifdef FP32
      // double precision
      memory->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE + i * 2] =
          w_q_lora_weights[i];
      memory->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE + i * 2 + 1] = 0;
#else
      Posit<16, 1> p16(w_q_lora_weights[i]);
      // double precision
      memory->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE + i * 2].setbits(
          p16.bits & 0xFF);
      memory->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE + i * 2 + 1]
          .setbits((p16.bits >> 8) & 0xFF);

#endif
    }

    std::string w_qb_lora_file =
        "models/mobilebert/binary_data/tiny_pretrained/step_" +
        std::to_string(stepNum) +
        "/weights/"
        "mobilebert_encoder_layer_" +
        std::to_string(encoderLayer) +
        "_attention_self_query_lora_B_default_weight";
    double *w_qb_lora_weights =
        readFileAsDouble(w_qb_lora_file, LORA_WQ_B_SIZE / 2, true);

    // B is stored as a transposed matrix
    for (int row = 0; row < 16; row++) {
      for (int col = 0; col < 128; col++) {
#ifdef FP32
        memory->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE +
                     LORA_WQ_A_SIZE + row * 128 * 2 + col * 2] =
            w_qb_lora_weights[col * 16 + row];
        memory->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE +
                     LORA_WQ_A_SIZE + row * 128 * 2 + col * 2 + 1] = 0;
#else
        Posit<16, 1> p16(w_qb_lora_weights[col * 16 + row]);
        memory
            ->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE +
                   LORA_WQ_A_SIZE + row * 128 * 2 + col * 2]
            .setbits(p16.bits & 0xFF);
        memory
            ->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE +
                   LORA_WQ_A_SIZE + row * 128 * 2 + col * 2 + 1]
            .setbits((p16.bits >> 8) & 0xFF);
#endif
      }
    }

    // load value LoRA weight
    std::string w_v_lora_file =
        "models/mobilebert/binary_data/tiny_pretrained/step_" +
        std::to_string(stepNum) +
        "/weights/"
        "mobilebert_encoder_layer_" +
        std::to_string(encoderLayer) +
        "_attention_self_value_lora_A_default_weight";
    double *w_v_lora_weights =
        readFileAsDouble(w_v_lora_file, LORA_WV_A_SIZE / 2, true);

    for (int i = 0; i < LORA_WV_A_SIZE / 2; i++) {
#ifdef FP32
      // double precision
      memory->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE +
                   LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + i * 2] =
          w_v_lora_weights[i];

      memory->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE +
                   LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + i * 2 + 1] = 0;
#else
      Posit<16, 1> p16(w_v_lora_weights[i]);
      // double precision
      memory
          ->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE + LORA_WQ_A_SIZE +
                 LORA_WQ_B_SIZE + i * 2]
          .setbits(p16.bits & 0xFF);
      memory
          ->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE + LORA_WQ_A_SIZE +
                 LORA_WQ_B_SIZE + i * 2 + 1]
          .setbits((p16.bits >> 8) & 0xFF);
#endif
    }

    std::string w_vb_lora_file =
        "models/mobilebert/binary_data/tiny_pretrained/step_" +
        std::to_string(stepNum) + "/weights/mobilebert_encoder_layer_" +
        std::to_string(encoderLayer) +
        "_attention_self_value_lora_B_default_weight";
    double *w_vb_lora_weights =
        readFileAsDouble(w_vb_lora_file, LORA_WV_B_SIZE / 2, true);

    // B is stored as a transposed matrix
    for (int row = 0; row < 16; row++) {
      for (int col = 0; col < 128; col++) {
#ifdef FP32
        memory->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE +
                     LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE +
                     row * 128 * 2 + col * 2] =
            w_vb_lora_weights[col * 16 + row];
        memory->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE +
                     LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE +
                     +row * 128 * 2 + col * 2 + 1] = 0;
#else
        Posit<16, 1> p16(w_vb_lora_weights[col * 16 + row]);
        memory
            ->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE +
                   LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE +
                   row * 128 * 2 + col * 2]
            .setbits(p16.bits & 0xFF);
        memory
            ->sram[LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE +
                   LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE +
                   row * 128 * 2 + col * 2 + 1]
            .setbits((p16.bits >> 8) & 0xFF);
#endif
      }
    }
  }

  // Initialize all LoRA weight gradients to 0
  for (int i = 0; i < NUM_ENCODER_LAYERS * LORA_W_PER_ENC_SIZE; i++) {
    memory->sram[LORA_G + i] = 0;
  }

  // Copy classifier weight to SRAM
  for (int i = 0; i < 16 * 512 * 2; i++) {
    int rramOffset = (NUM_ENCODER_LAYERS - 1) * ENCODER_WEIGHT_SIZE +
                     8 * INTERMEDIATE_SIZE + 5 * INTERMEDIATE_BIAS_SIZE +
                     3 * INTRA_BOTTLENECK_SIZE +
                     18 * INTRA_BOTTLENECK_BIAS_SIZE;
    memory->sram[CLASSIFIER_W + i] = memory->rram[rramOffset + i];
  }

  // Copy classifier bias to SRAM
  for (int i = 0; i < 16 * 2; i++) {
    int rramOffset = (NUM_ENCODER_LAYERS - 1) * ENCODER_WEIGHT_SIZE +
                     8 * INTERMEDIATE_SIZE + 21 * INTERMEDIATE_BIAS_SIZE +
                     3 * INTRA_BOTTLENECK_SIZE +
                     18 * INTRA_BOTTLENECK_BIAS_SIZE;

    memory->sram[CLASSIFIER_B + i] = memory->rram[rramOffset + i];
  }

  // initialize classifier weight gradient to 0
  for (int i = 0; i < CLASSIFIER_G_SIZE; i++) {
    memory->sram[CLASSIFIER_G + i] = 0;
  }

  // initialize classifier bias gradient to 0
  for (int i = 0; i < CLASSIFIER_B_SIZE; i++) {
    memory->sram[CLASSIFIER_B_G + i] = 0;
  }
}

int main(int argc, char **argv) {
  memory = new SimpleMemoryModel<DTYPE>(false);
  initialize_model("models/mobilebert/binary_data/tiny_pretrained/");

#ifdef MRPC
  DatasetIterator trainingSet(
      "models/mobilebert/binary_data/tiny_mrpc_finetuning/train/");
  DatasetIterator validationSet(
      "models/mobilebert/binary_data/tiny_mrpc_finetuning/validation/");
#else
  DatasetIterator dataset("models/mobilebert/binary_data/tiny_pretrained/");
#endif

  for (int i = 0; i < 100; i++) {
#ifdef MRPC
    load_sample(trainingSet, false, i);
#else
    load_sample(dataset, true, i);
#endif

    full_forward_pass();
    full_backward_pass(i);
  }
}
