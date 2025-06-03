/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "yolo_model.h"
#include "main_functions.h"
#include "constants.h"
#include "output_handler.h"

#include "esp_heap_caps.h"
#include "esp_log.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;
  int inference_count = 0;

  /* old
  constexpr int kTensorArenaSize = 2000;
  uint8_t tensor_arena[kTensorArenaSize];
  */
  // new
  constexpr int kTensorArenaSize = 200 * 1024; // Adjust as needed
  uint8_t *tensor_arena = (uint8_t *)heap_caps_malloc(
      kTensorArenaSize,
      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

} // namespace

// The name of this function is important for Arduino compatibility.
void setup()
{
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  // old
  // model = tflite::GetModel(yolo_model);
  // new
  // area is ok
  if (!tensor_arena)
  {
    ESP_LOGE("TFLITE", "Failed to allocate tensor arena in PSRAM");
    return;
  }

  const char *TAG = "TFLITE";

  uint8_t *model_in_psram = (uint8_t *)heap_caps_malloc(yolo_model_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  if (!model_in_psram)
  {
    ESP_LOGE(TAG, "Failed to allocate PSRAM for model");
    return;
  }

  memcpy(model_in_psram, yolo_model, yolo_model_len);

  // Now pass to TFLM
  model = tflite::GetModel(model_in_psram);

  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk)
  {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void loop()
{
  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  float position = static_cast<float>(inference_count) /
                   static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  // Quantize the input from floating-point to integer
  int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  // Place the quantized input in the model's input tensor
  input->data.int8[0] = x_quantized;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk)
  {
    MicroPrintf("Invoke failed on x: %f\n",
                static_cast<double>(x));
    return;
  }

  // Obtain the quantized output from model's output tensor
  int8_t y_quantized = output->data.int8[0];
  // Dequantize the output from integer to floating-point
  float y = (y_quantized - output->params.zero_point) * output->params.scale;

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(x, y);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle)
    inference_count = 0;
}
