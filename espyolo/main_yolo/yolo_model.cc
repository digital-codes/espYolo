extern "C" {
#include "yolo_model.h"  // include it for consistency

alignas(8) const unsigned char yolo_model[] = {
#include "yolo_model.inc"
};

const unsigned int yolo_model_len = sizeof(yolo_model);
}
