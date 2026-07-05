# -----------------------------------------------------------------------------
# AC runtime estimation from indoor temperature/humidity sensor data
# -----------------------------------------------------------------------------
# This script trains deep-learning classifiers for AC on/off estimation using
# windowed ambient sensor data. It is intentionally organised as a reproducible
# experiment runner: change the configuration block below, then run the script.
# -----------------------------------------------------------------------------

# ----------------------------- Environment setup -----------------------------
# Keep these CUDA settings if the project is run on the same GPU server. If the
# code is used on another machine, update or remove these paths as needed.
Sys.unsetenv("CUDA_VISIBLE_DEVICES")
Sys.setenv(PATH = "/usr/local/cuda-12.8/bin:/usr/bin:/bin")
Sys.setenv(LD_LIBRARY_PATH = "/usr/local/cuda-12.8/lib64")
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "0")
Sys.setenv(TF_ENABLE_ONEDNN_OPTS = "0")
Sys.setenv(CUDA_VISIBLE_DEVICES = "0")

suppressPackageStartupMessages({
  library(tidyverse)
  library(keras3)
  library(tensorflow)
  library(zoo)
  library(rsample)
})

source("ac_helper_code.R")

# ------------------------------- Configuration --------------------------------
CODE <- "Ac"
NUMBER_OF_CLASSES <- 2
AGGREGATION_MINUTES <- 1
EPOCHS <- 5
REQUIRE_GPU <- TRUE

# Use an existing cleaned file when available; otherwise build it from raw CSVs.
CLEAN_SENSOR_FILE <- "./data/ac_sensor_data/all_sensor_data.csv"

# Feature-set options used in the experiments.
FEATURE_OPTIONS <- c(
  temp = "temp",
  humi = "humi",
  temp_humi = "temp_humi"
)

# Experiment grid. The original run used temperature only, 180/210-minute
# windows, and a one-minute stride (`overlap = time_steps - 1`).
SELECTED_FEATURES <- c("temp")
TIME_STEPS_GRID <- c(180, 210)
OVERLAP_SETTINGS <- c(1)
MODEL_IDS <- c(2) # 1 = GTN, 2 = bidirectional LSTM, 3 = 1D CNN
ROOM_IDS_FOR_FILENAME <- c(328)

MODEL_LOSS <- if (NUMBER_OF_CLASSES == 2) "binary_crossentropy" else "categorical_crossentropy"
MODEL_ACTIVATION <- if (NUMBER_OF_CLASSES == 2) "sigmoid" else "softmax"

# ------------------------------- Data loading ---------------------------------
if (file.exists(CLEAN_SENSOR_FILE)) {
  sensor_data <- read.csv(file = CLEAN_SENSOR_FILE)
  message("Loaded cleaned sensor file: ", CLEAN_SENSOR_FILE)
} else {
  sensor_data <- load_data(
    code = CODE,
    dataset_type = "train",
    interpolate = TRUE,
    number_of_classes = NUMBER_OF_CLASSES
  )
}

message("Sensor data dimensions: ", paste(dim(sensor_data), collapse = " x "))

# ------------------------------ Hardware check --------------------------------
start_time <- Sys.time()
tf_config <- tensorflow::tf$config$list_physical_devices("GPU")
verify_time <- as.numeric(Sys.time() - start_time, units = "secs")

if (length(tf_config) > 0) {
  message("GPU detected after ", round(verify_time, 2), " seconds:")
  print(tf_config)
} else if (REQUIRE_GPU) {
  stop("GPU not detected. Set REQUIRE_GPU <- FALSE to allow CPU execution.")
} else {
  warning("GPU not detected. Continuing on CPU.")
}

#' Build one of the supported AC classification models.
#'
#' @param model_id Numeric model identifier: 1 = GTN, 2 = bidirectional LSTM,
#'   3 = 1D CNN.
#' @param time_steps Number of observations in each input window.
#' @param number_of_features Number of sensor feature channels.
#' @param number_of_classes Number of target classes.
#' @param model_activation Output activation function.
#' @return A list with the uncompiled Keras model and a model-name suffix.
build_model <- function(model_id,
                        time_steps,
                        number_of_features,
                        number_of_classes,
                        model_activation) {
  if (model_id == 1) {
    return(list(
      model = create_GTN(
        time_steps = time_steps,
        number_of_features = number_of_features,
        number_of_classes = number_of_classes,
        model_activation = model_activation
      ),
      suffix = "Ml(GTN)"
    ))
  }

  if (model_id == 2) {
    model <- keras_model_sequential() %>%
      layer_dense(units = 8, input_shape = c(time_steps, number_of_features)) %>%
      bidirectional(layer_lstm(units = 16, return_sequences = TRUE)) %>%
      layer_dropout(0.2) %>%
      bidirectional(layer_lstm(units = 16, return_sequences = TRUE)) %>%
      layer_dropout(0.2) %>%
      layer_flatten() %>%
      layer_dense(units = number_of_classes, activation = model_activation)

    return(list(
      model = model,
      suffix = "Ml(dens8-bid16-drop.2-bid16-drop.2-flat-out)"
    ))
  }

  if (model_id == 3) {
    model <- keras_model_sequential() %>%
      layer_conv_1d(
        filters = 2,
        kernel_size = 3,
        activation = "relu",
        input_shape = c(time_steps, number_of_features)
      ) %>%
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu") %>%
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_flatten() %>%
      layer_dropout(rate = 0.5) %>%
      layer_dense(units = 128, activation = "relu") %>%
      layer_dense(units = number_of_classes, activation = model_activation)

    return(list(
      model = model,
      suffix = "Ml(cnn2_3-pool2-cnn32_3-pool2-flat-drop.5-dens128-out5)"
    ))
  }

  stop("Unsupported MODEL_ID: ", model_id)
}

# ------------------------------- Training loop --------------------------------
for (model_id in MODEL_IDS) {
  tryCatch({
    for (selected_feature in SELECTED_FEATURES) {
      tryCatch({
        number_of_features <- get_number_of_features(selected_feature)

        for (overlap_setting in OVERLAP_SETTINGS) {
          tryCatch({
            for (time_steps in TIME_STEPS_GRID) {
              tryCatch({
                overlap <- if (overlap_setting == 1) {
                  time_steps - 1
                } else {
                  floor(time_steps * overlap_setting)
                }

                message("Preparing data: feature=", selected_feature,
                        ", time_steps=", time_steps,
                        ", overlap=", overlap)

                prepared_data <- prepare_data(
                  sensor_data = sensor_data,
                  features = selected_feature,
                  time_steps = time_steps,
                  overlap = overlap,
                  type = "train",
                  include_test = TRUE,
                  out_one_room = FALSE,
                  number_of_classes = NUMBER_OF_CLASSES
                )

                for (room_id_for_filename in ROOM_IDS_FOR_FILENAME) {
                  tryCatch({
                    message(
                      "Model=", model_id,
                      " | feature=", selected_feature,
                      " | window=", time_steps,
                      " | overlap=", overlap
                    )

                    X_train <- prepared_data$X_train
                    y_train <- prepared_data$y_train
                    X_val <- prepared_data$X_val
                    y_val <- prepared_data$y_val
                    X_test <- prepared_data$X_test
                    y_test <- prepared_data$y_test

                    tensorflow::set_random_seed(10)

                    model_details <- paste0("_RO", room_id_for_filename, "_")
                    model_bundle <- build_model(
                      model_id = model_id,
                      time_steps = time_steps,
                      number_of_features = number_of_features,
                      number_of_classes = NUMBER_OF_CLASSES,
                      model_activation = MODEL_ACTIVATION
                    )
                    model <- model_bundle$model
                    model_details <- paste0(model_details, model_bundle$suffix)

                    model %>% compile(
                      loss = MODEL_LOSS,
                      optimizer = optimizer_adam(),
                      metrics = c("accuracy", "recall", "precision")
                    )

                    early_stopping_callback <- callback_early_stopping(
                      monitor = "val_loss",
                      patience = 3,
                      restore_best_weights = TRUE,
                      mode = "min"
                    )

                    training_start_time <- Sys.time()
                    train_val_result <- model %>% fit(
                      x = X_train,
                      y = y_train,
                      epochs = EPOCHS,
                      validation_data = list(X_val, y_val),
                      verbose = 2,
                      callbacks = list(early_stopping_callback)
                    )

                    test_results <- model %>% evaluate(X_test, y_test, verbose = 0)
                    training_time <- as.numeric(Sys.time() - training_start_time, units = "secs")

                    file_name <- store_test_as_csv(
                      metrics = test_results,
                      train_val_result = train_val_result,
                      model_string = model_details,
                      train_size = dim(X_train)[1],
                      val_size = dim(X_val)[1],
                      test_size = dim(X_test)[1],
                      training_time = training_time,
                      epochs_num = length(train_val_result$metrics$loss),
                      code = CODE,
                      selected_feature = selected_feature,
                      number_of_classes = NUMBER_OF_CLASSES,
                      aggregate = AGGREGATION_MINUTES,
                      time_steps = time_steps,
                      overlap = overlap,
                      sensor_data_rows = nrow(sensor_data),
                      model_details = model_details
                    )

                    save_my_model(file_name, model)

                    rm(
                      X_train, X_val, X_test,
                      y_train, y_val, y_test,
                      model, train_val_result, test_results
                    )
                    tensorflow::tf$keras$backend$clear_session()
                    gc()
                  }, error = function(e) {
                    message("Error in room/file-name loop (room_id = ",
                            room_id_for_filename, "): ", e$message)
                  })
                }
              }, error = function(e) {
                message("Error in time-step loop (time_steps = ", time_steps, "): ", e$message)
              })
            }
          }, error = function(e) {
            message("Error in overlap loop (overlap_setting = ", overlap_setting, "): ", e$message)
          })
        }
      }, error = function(e) {
        message("Error in feature loop (selected_feature = ", selected_feature, "): ", e$message)
      })
    }
  }, error = function(e) {
    message("Error in model loop (model_id = ", model_id, "): ", e$message)
  })
}

message("Done!")
