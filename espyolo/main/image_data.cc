extern "C" {
#include "image_data.h"  // include it for consistency
#include "constants.h"

//const int imgSize = yoloWidth * yoloHeight * 3;

const unsigned char image_data[] = {
#include "robot_204_data.inc"
};


}
