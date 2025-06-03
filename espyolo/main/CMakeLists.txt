set(SOURCES
    main.cc main_functions.cc constants.cc output_handler.cc yolo_model.cc 
)

idf_component_register(SRCS ${SOURCES}
    PRIV_REQUIRES spi_flash
    INCLUDE_DIRS ".")

