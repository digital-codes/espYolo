/**
 * This example takes a picture every 5s and print its size on serial monitor.
 */

// =============================== SETUP ======================================

// 1. Board setup (Uncomment):
// #define BOARD_WROVER_KIT
// #define BOARD_ESP32CAM_AITHINKER
// #define BOARD_ESP32S3_WROOM

/**
 * 2. Kconfig setup
 *
 * If you have a Kconfig file, copy the content from
 *  https://github.com/espressif/esp32-camera/blob/master/Kconfig into it.
 * In case you haven't, copy and paste this Kconfig file inside the src directory.
 * This Kconfig file has definitions that allows more control over the camera and
 * how it will be initialized.
 */

/**
 * 3. Enable PSRAM on sdkconfig:
 *
 * CONFIG_ESP32_SPIRAM_SUPPORT=y
 *
 * More info on
 * https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/kconfig.html#config-esp32-spiram-support
 */

// ================================ CODE ======================================

#include <esp_log.h>
#include <esp_system.h>
#include <nvs_flash.h>
#include <sys/param.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// support IDF 5.x
#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif

#include "esp_camera.h"
#include "esp_wifi.h"
#include "lwip/sockets.h"

#define PORT 3333  // choose a port

//#define BOARD_WROVER_KIT 1

#define BOARD_ESP32S3RCAM

#ifdef BOARD_ESP32S3RCAM
#define CAM_PIN_PWDN  -1
#define CAM_PIN_RESET -1
#define CAM_PIN_XCLK  21
#define CAM_PIN_SIOD  12
#define CAM_PIN_SIOC  9
#define CAM_PIN_D7    13
#define CAM_PIN_D6    11
#define CAM_PIN_D5    17
#define CAM_PIN_D4    4
#define CAM_PIN_D3    48
#define CAM_PIN_D2    46
#define CAM_PIN_D1    42
#define CAM_PIN_D0    3
#define CAM_PIN_VSYNC 10
#define CAM_PIN_HREF  14
#define CAM_PIN_PCLK  40

#define POWER_GPIO_NUM 18
#endif 

static const char *TAG = "example:take_picture";

static camera_config_t camera_config = {
    .pin_pwdn = CAM_PIN_PWDN,
    .pin_reset = CAM_PIN_RESET,
    .pin_xclk = CAM_PIN_XCLK,
    .pin_sccb_sda = CAM_PIN_SIOD,
    .pin_sccb_scl = CAM_PIN_SIOC,

    .pin_d7 = CAM_PIN_D7,
    .pin_d6 = CAM_PIN_D6,
    .pin_d5 = CAM_PIN_D5,
    .pin_d4 = CAM_PIN_D4,
    .pin_d3 = CAM_PIN_D3,
    .pin_d2 = CAM_PIN_D2,
    .pin_d1 = CAM_PIN_D1,
    .pin_d0 = CAM_PIN_D0,
    .pin_vsync = CAM_PIN_VSYNC,
    .pin_href = CAM_PIN_HREF,
    .pin_pclk = CAM_PIN_PCLK,

    //XCLK 20MHz or 10MHz for OV2640 double FPS (Experimental)
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_RGB565, //YUV422,GRAYSCALE,RGB565,JPEG
    .frame_size = FRAMESIZE_QCIF,    //QCIF 176x144 works. QVGA 320*240 works. 128X128 is probably QQVGA (160*120)   QQVGA-UXGA, For ESP32, do not use sizes above QVGA when not JPEG. The performance of the ESP32-S series has improved a lot, but JPEG mode always gives better frame rates.

    .jpeg_quality = 12, //0-63, for OV series camera sensors, lower number means higher quality
    .fb_count = 1, // 2,       //When jpeg mode is used, if fb_count more than one, the driver will work in continuous mode.
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY //CAMERA_GRAB_LATEST, // CAMERA_GRAB_WHEN_EMPTY,
};

static esp_err_t init_camera(void)
{
    //initialize the camera
    ESP_LOGI(TAG, "Cam init...");

    ESP_LOGI(TAG, "ESP32S3RCAM...");
    gpio_config_t io_conf = {
        .pin_bit_mask = 1ULL << POWER_GPIO_NUM,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);

    gpio_set_level(POWER_GPIO_NUM, 1);
    vTaskDelay(1000 / portTICK_RATE_MS);
    gpio_set_level(POWER_GPIO_NUM, 0);
    vTaskDelay(1000 / portTICK_RATE_MS);


    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK)
    {
        ESP_LOGE(TAG, "Camera Init Failed");
        return err;
    }

    return ESP_OK;
}

void do_wifi(void *pvParameters);
void image_socket_server_task(void *pvParameters);

void app_main(void)
{
    xTaskCreate((TaskFunction_t)do_wifi, "WiFiTask", 4096, NULL, 5, NULL);
    ESP_LOGI(TAG, "Wifi initialized successfully");

/*
https://github.com/espressif/esp32-camera/issues/314
https://forum.arduino.cc/t/esp32-cam-ov2640-dark-picture/1202399
*/

    if(ESP_OK != init_camera()) {
        return;
    }

    ESP_LOGI(TAG, "Camera initialized successfully");
    /*
    sensor_t *s = esp_camera_sensor_get();
    ESP_LOGI(TAG, "GAIN CTRL");
    s->set_gain_ctrl(s, 0x10);
    vTaskDelay(100 / portTICK_RATE_MS);
    */
   
    const int grabDelayMs = 500;
    vTaskDelay(grabDelayMs / portTICK_RATE_MS);
    xTaskCreate((TaskFunction_t)image_socket_server_task, "SendImage", 4096, NULL, 5, NULL);

    while (1)
    {
        ESP_LOGI(TAG, "Taking picture...");
        #if 0
        camera_fb_t *pic = esp_camera_fb_get();

        // use pic->buf to access the image
        ESP_LOGI(TAG, "Picture taken! Its size was: %zu bytes", pic->len);

        // Compute average intensity by color
        float sum_red = 0, sum_green = 0, sum_blue = 0;
        uint32_t pixel_count = 0;
        // Assuming the pixel format is RGB565
        // Each pixel is represented by 2 bytes in RGB565 format
        if (pic->format != PIXFORMAT_RGB565) {
            ESP_LOGE(TAG, "Unsupported pixel format: %d", pic->format);
            esp_camera_fb_return(pic);
            vTaskDelay(grabDelayMs / portTICK_RATE_MS);
            continue;
        }
        if (pic->len % 2 != 0) {
            ESP_LOGE(TAG, "Picture length is not a multiple of 2, cannot process RGB565 format");
            esp_camera_fb_return(pic);
            vTaskDelay(grabDelayMs / portTICK_RATE_MS);
            continue;
        }
        if (pic->len != 176*144*2 ) {  // QCIF
            ESP_LOGE(TAG, "Picture length is not QCIF");
            esp_camera_fb_return(pic);
            vTaskDelay(grabDelayMs / portTICK_RATE_MS);
            continue;
        }
        ESP_LOGI(TAG, "Processing picture for average intensity...");

        for (size_t i = 0; i < pic->len; i += 2) {
            uint16_t pixel = pic->buf[i] | (pic->buf[i + 1] << 8);
            uint8_t red = (pixel >> 11) & 0x1F;
            uint8_t green = (pixel >> 5) & 0x3F;
            uint8_t blue = pixel & 0x1F;

            sum_red += (float)red;
            sum_green += (float)green;
            sum_blue += (float)blue;
            pixel_count++;
        }
        ESP_LOGI(TAG, "Intensity sums - Red: %.2f, Green: %.2f, Blue: %.2f", sum_red, sum_green, sum_blue);

        float avg_red = (float)(sum_red / pixel_count);
        float avg_green = (float)(sum_green / pixel_count);
        float avg_blue = (float)(sum_blue / pixel_count);

        ESP_LOGI(TAG, "Average Intensity - Red: %.2f, Green: %.2f, Blue: %.2f", avg_red, avg_green, avg_blue);

        esp_camera_fb_return(pic);
        #endif
        vTaskDelay(grabDelayMs / portTICK_RATE_MS);
    }
}



void image_socket_server_task(void *pvParameters) {
    int listen_sock, client_sock;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);

    listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    bind(listen_sock, (struct sockaddr *)&server_addr, sizeof(server_addr));
    listen(listen_sock, 1);

    client_sock = accept(listen_sock, (struct sockaddr *)&client_addr, &client_addr_len);
    while (1) {

        // Capture image
        camera_fb_t *pic = esp_camera_fb_get();
        if (pic) {
            // use pic->buf to access the image
            ESP_LOGI(TAG, "Picture taken! Its size was: %zu bytes", pic->len);

            // First send length
            uint32_t len = pic->len;
            uint32_t len_network_order = htonl(len);
            send(client_sock, &len_network_order, sizeof(len_network_order), 0);
            // Send image in 2KB chunks
            size_t bytes_sent = 0;
            const size_t chunk_size = 2048;
            while (bytes_sent < pic->len) {
                size_t bytes_to_send = (pic->len - bytes_sent > chunk_size) ? chunk_size : (pic->len - bytes_sent);
                send(client_sock, pic->buf + bytes_sent, bytes_to_send, 0);
                bytes_sent += bytes_to_send;
            }
            //send(client_sock, pic->buf, pic->len, 0);
            esp_camera_fb_return(pic);
        }

        const int grabDelayMs = 200;
        vTaskDelay(grabDelayMs / portTICK_RATE_MS);
    }
    close(client_sock);  // close after one image, or loop to serve more
}
