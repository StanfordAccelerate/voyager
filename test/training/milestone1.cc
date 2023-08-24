// Milestone 1- backward pass only, with loading of activations
void milestone1(SimpleMemoryModel<DTYPE> *memory) {
  MobileBERT mobilebert("mobilebert", "backward",
                        "models/mobilebert/binary_data/tiny_pretrained/");

  std::vector<Workload> backwardPass = mobilebert.getFullBackwardPass();
  bool flag = false;
  int memoryWatch = 0;
  DTYPE memoryWatchVal = 0;
  for (int stepNumber = 0; stepNumber < 1; stepNumber++) {
    for (Workload &workload : backwardPass) {
      std::cout << "Running " << workload.name << std::endl;

      // adjust all files to point to the correct step number
      adjustPathForStep(workload.files.inputs_file, stepNumber);
      adjustPathForStep(workload.files.weights_file, stepNumber);
      adjustPathForStep(workload.files.bias_file, stepNumber);
      adjustPathForStep(workload.files.outputs_file, stepNumber);
      adjustPathForStep(workload.files.residual_file, stepNumber);

      // remove file entry if it doesn't have "activations" in it
      if (workload.files.inputs_file.find("activations") == std::string::npos) {
        workload.files.inputs_file = "";
      }
      if (workload.files.weights_file.find("activations") ==
          std::string::npos) {
        workload.files.weights_file = "";
      }
      if (workload.files.bias_file.find("activations") == std::string::npos) {
        workload.files.bias_file = "";
      }
      if (workload.files.residual_file.find("activations") ==
          std::string::npos) {
        workload.files.residual_file = "";
      }

      memory->loadModelActivations(workload.params, workload.files,
                                   workload.memoryMap, true);

      runWorkload(memory, workload);
      checkOutputs(memory, workload);
    }
  }
}