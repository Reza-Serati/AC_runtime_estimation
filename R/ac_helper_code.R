# -----------------------------------------------------------------------------
# Helper functions for AC runtime estimation from ambient sensor data
# -----------------------------------------------------------------------------
# This file contains reusable functions for:
#   1. loading and cleaning room/outdoor sensor CSV files;
#   2. interpolating irregular observations to a regular time grid;
#   3. converting continuous time series into overlapping model windows;
#   4. saving model metrics and trained Keras models; and
#   5. running predictions and summarising model performance.
#
# The training script (`ac_estimation.R`) loads the required packages and sources
# this helper file. Keep project-specific paths in `load_data()` or pass cleaned
# data frames directly into `prepare_data()` when adapting the code.
# -----------------------------------------------------------------------------

#' Create a directory if it does not already exist.
#'
#' @param path Directory path to create.
#' @return Invisibly returns TRUE if the directory exists after the call.
ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  invisible(dir.exists(path))
}

#' Load and clean AC sensor data from CSV files.
#'
#' For AC data, room files are expected under `./data/ac_sensor_data/` and must
#' follow the pattern `sensor*.csv`. Outdoor files are expected under
#' `./data/ac_sensor_data/Outside` and must follow the pattern `outside*.csv`.
#'
#' Expected room columns: `X.timestamp`, `sensor.temperature`,
#' `sensor.humidity`, and `label`. Expected outdoor columns: `X.timestamp`,
#' `sensor.temperature`, and `sensor.humidity`.
#'
#' @param code Dataset code. Currently supports `"Ac"` and the legacy `"Shower"`
#'   folder structure.
#' @param dataset_type Either `"train"` for room data or `"outside"` for outdoor
#'   ambient data.
#' @param interpolate Logical. If TRUE, each file is interpolated to a regular
#'   one-minute time grid.
#' @param apply_filter Logical. If TRUE, applies a smoothing filter to temperature
#'   and humidity.
#' @param filter_type Smoothing filter. Supports `"moving_average"` and
#'   `"median"`.
#' @param filter_window Window length for the selected smoothing filter.
#' @param number_of_classes Number of target classes. If 2, non-zero labels are
#'   converted to 1.
#' @return A single cleaned data frame containing all matching files.
load_data <- function(code = "Ac",
                      dataset_type = c("train", "outside"),
                      interpolate = FALSE,
                      apply_filter = FALSE,
                      filter_type = c("moving_average", "median"),
                      filter_window = 5,
                      number_of_classes = 2) {
  dataset_type <- match.arg(dataset_type)
  filter_type <- match.arg(filter_type)

  if (code == "Ac") {
    directory_path <- if (dataset_type == "outside") {
      "./data/ac_sensor_data/Outside"
    } else {
      "./data/ac_sensor_data/"
    }
  } else if (code == "Shower") {
    directory_path <- "./data/shower_sensor_data/"
  } else {
    stop("Unsupported dataset code: ", code)
  }

  file_pattern <- if (dataset_type == "outside") {
    "^outside.*\\.csv$"
  } else {
    "^sensor.*\\.csv$"
  }

  file_list <- list.files(
    path = directory_path,
    pattern = file_pattern,
    full.names = TRUE
  )

  if (length(file_list) == 0) {
    stop("No matching CSV files found in: ", directory_path)
  }

  data_list <- list()

  for (file_path in file_list) {
    message("Loading: ", file_path)
    temp_data <- read.csv(file_path)
    expected_columns <- if (dataset_type == "outside") 3 else 4

    if (ncol(temp_data) != expected_columns) {
      warning("Skipping file with unexpected number of columns: ", file_path)
      next
    }

    temp_data <- temp_data %>%
      dplyr::filter(
        !is.na(suppressWarnings(as.numeric(as.character(sensor.humidity)))) &
          !is.na(suppressWarnings(as.numeric(as.character(sensor.temperature))))
      )

    temp_data$X.timestamp <- lubridate::parse_date_time(
      temp_data$X.timestamp,
      orders = "ymd HMS"
    )
    temp_data$sensor.temperature <- as.numeric(as.character(temp_data$sensor.temperature))
    temp_data$sensor.humidity <- as.numeric(as.character(temp_data$sensor.humidity))

    if (dataset_type == "outside") {
      temp_data <- temp_data %>% dplyr::mutate(sensor.number = "outside")
    } else {
      sensor_number <- as.numeric(sub(".*sensor(\\d+)_.*\\.csv$", "\\1", file_path))
      temp_data <- temp_data %>% dplyr::mutate(sensor.number = sensor_number)

      temp_data <- temp_data %>%
        dplyr::filter(!is.na(suppressWarnings(as.numeric(as.character(label)))))
      temp_data$label <- as.numeric(temp_data$label)

      if (number_of_classes == 2) {
        temp_data$label <- ifelse(temp_data$label != 0, 1, 0)
      }
    }

    if (nrow(temp_data) > 1 && interpolate) {
      temp_data <- interpolate_data(temp_data, time_interval = "1 min")
    }

    if (apply_filter) {
      if (filter_type == "moving_average") {
        temp_data <- temp_data %>%
          dplyr::mutate(
            sensor.temperature = zoo::rollmean(
              sensor.temperature,
              filter_window,
              fill = NA,
              align = "center"
            ),
            sensor.humidity = zoo::rollmean(
              sensor.humidity,
              filter_window,
              fill = NA,
              align = "center"
            )
          )
      } else if (filter_type == "median") {
        temp_data <- temp_data %>%
          dplyr::mutate(
            sensor.temperature = stats::runmed(sensor.temperature, filter_window),
            sensor.humidity = stats::runmed(sensor.humidity, filter_window)
          )
      }
    }

    data_list[[file_path]] <- temp_data
  }

  if (length(data_list) == 0) {
    stop("No valid files were loaded from: ", directory_path)
  }

  dplyr::bind_rows(data_list)
}

#' Return the number of model input features for a feature set name.
#'
#' @param selected_feature One of `"temp"`, `"humi"`, or `"temp_humi"`.
#' @return Integer number of feature channels.
get_number_of_features <- function(selected_feature) {
  if (selected_feature == "temp") {
    return(1)
  }
  if (selected_feature == "humi") {
    return(1)
  }
  if (selected_feature == "temp_humi") {
    return(2)
  }

  stop("Unknown feature selection: ", selected_feature)
}

#' Aggregate a vector pair over fixed windows.
#'
#' This utility is kept for compatibility with earlier experiments. With the
#' default `aggrigate = 1`, it returns the input vectors unchanged.
#'
#' @param X Feature vector.
#' @param Y Label vector aligned with `X`.
#' @param aggrigate Window length. Kept with original spelling for backwards
#'   compatibility.
#' @param overlap Number of overlapping observations between aggregation windows.
#' @return A list with aggregated `X` and `Y` vectors.
aggregate_data <- function(X, Y, aggrigate = 1, overlap = 0) {
  if (aggrigate <= 1) {
    return(list(X = X, Y = Y))
  }

  step <- aggrigate - overlap
  if (step <= 0) {
    stop("Aggregation window must be larger than overlap.")
  }

  final_x <- c()
  final_y <- c()
  for (i in seq(1, length(X) - aggrigate + 1, by = step)) {
    final_x <- c(final_x, mean(X[i:(i + aggrigate - 1)], na.rm = TRUE))
    final_y <- c(final_y, mean(Y[i:(i + aggrigate - 1)], na.rm = TRUE))
  }

  list(X = final_x, Y = final_y)
}

#' Convert integer class labels to a one-hot encoded matrix.
#'
#' @param y Integer class labels starting at 0.
#' @param num_classes Number of output classes. If NULL, inferred from `y`.
#' @return A one-hot encoded matrix with one row per label.
to_categorical_custom <- function(y, num_classes = NULL) {
  if (is.null(num_classes)) {
    num_classes <- length(unique(y))
  }

  categorical_matrix <- matrix(0, nrow = length(y), ncol = num_classes)

  for (i in seq_along(y)) {
    class_label <- y[i]
    if (is.na(class_label)) {
      next
    }
    categorical_matrix[i, class_label + 1] <- 1
  }

  categorical_matrix
}

#' Interpolate a sensor data frame to a regular time grid.
#'
#' Temperature and humidity are linearly interpolated. The sensor number and label
#' columns, when present, are carried forward.
#'
#' @param data Sensor data frame containing `X.timestamp`, `sensor.temperature`,
#'   and `sensor.humidity`.
#' @param time_interval Time step used in `seq()`, for example `"1 min"`.
#' @return Data frame on a regular timestamp grid.
interpolate_data <- function(data, time_interval = "1 min") {
  data <- data %>%
    dplyr::mutate(X.timestamp = as.POSIXct(X.timestamp, format = "%Y-%m-%d a %H:%M")) %>%
    dplyr::mutate(X.timestamp = as.POSIXct(format(X.timestamp, "%Y-%m-%d %H:%M:00"))) %>%
    dplyr::arrange(X.timestamp)

  full_timestamps <- seq(min(data$X.timestamp), max(data$X.timestamp), by = time_interval)
  full_data <- data.frame(X.timestamp = full_timestamps)

  full_data <- full_data %>% dplyr::left_join(data, by = "X.timestamp")
  full_data$sensor.temperature <- zoo::na.approx(full_data$sensor.temperature, na.rm = FALSE)
  full_data$sensor.humidity <- zoo::na.approx(full_data$sensor.humidity, na.rm = FALSE)
  full_data$sensor.number <- zoo::na.locf(full_data$sensor.number, na.rm = FALSE)

  if ("label" %in% colnames(full_data)) {
    full_data$label <- zoo::na.locf(full_data$label, na.rm = FALSE)
  }

  full_data
}

#' Min-max normalise a numeric vector.
#'
#' @param x Numeric vector.
#' @return Vector scaled to the [0, 1] range. Constant vectors return zeros.
min_max_normalize <- function(x) {
  range_x <- range(x, na.rm = TRUE)
  if (diff(range_x) == 0) {
    return(rep(0, length(x)))
  }
  (x - range_x[1]) / diff(range_x)
}

#' Convert room sensor data into overlapping 3D model windows.
#'
#' The returned `X` array has shape `[samples, time_steps, features]`. For
#' training, labels are one-hot encoded. For prediction/test modes, only the
#' input array is returned.
#'
#' @param sensor_data Cleaned room sensor data.
#' @param features Feature set: `"temp"`, `"humi"`, or `"temp_humi"`.
#' @param time_steps Number of observations in each input window.
#' @param overlap Number of overlapping observations between adjacent windows.
#' @param type Either `"train"`, `"test"`, or `"bar_whisker"`.
#' @param include_test Logical. If TRUE, creates train/validation/test splits.
#' @param normalize_data Logical. If TRUE, min-max normalises each sensor channel.
#' @param out_one_room Optional room/sensor number to exclude from training.
#' @param number_of_classes Number of target classes for one-hot encoding.
#' @param use_cache Logical. If TRUE, train/validation/test arrays are cached as
#'   RDS files under `./data/ac_sensor_data/Sampled/`.
#' @return A list of arrays for training mode or an input array for prediction.
prepare_data <- function(sensor_data,
                         features,
                         time_steps,
                         overlap,
                         type = c("train", "test", "bar_whisker"),
                         include_test = FALSE,
                         normalize_data = FALSE,
                         out_one_room = FALSE,
                         number_of_classes = 2,
                         use_cache = TRUE) {
  type <- match.arg(type)

  if (time_steps <= overlap) {
    stop("`time_steps` must be larger than `overlap`.")
  }

  folder_name <- file.path(
    "./data/ac_sensor_data/Sampled",
    features,
    paste0("WS", time_steps, "Ol", overlap)
  )

  if (type == "train" && include_test && use_cache && dir.exists(folder_name)) {
    rds_files <- file.path(
      folder_name,
      c("X_train.rds", "y_train.rds", "X_val.rds", "y_val.rds", "X_test.rds", "y_test.rds")
    )

    if (all(file.exists(rds_files))) {
      return(list(
        X_train = readRDS(rds_files[1]),
        y_train = readRDS(rds_files[2]),
        X_val = readRDS(rds_files[3]),
        y_val = readRDS(rds_files[4]),
        X_test = readRDS(rds_files[5]),
        y_test = readRDS(rds_files[6])
      ))
    }
  }

  if (type == "train") {
    data <- if (identical(out_one_room, FALSE)) {
      sensor_data
    } else {
      sensor_data %>% dplyr::filter(sensor.number != out_one_room)
    }
  } else {
    data <- sensor_data
    if (!"label" %in% colnames(data)) {
      data$label <- 0
    }
  }

  X_inside_temp <- round(data$sensor.temperature, 2)
  X_inside_humidity <- round(data$sensor.humidity, 2)
  Y <- data$label

  aggregated_temp <- aggregate_data(X_inside_temp, Y, aggrigate = 1)
  aggregated_humidity <- aggregate_data(X_inside_humidity, Y, aggrigate = 1)

  X_inside_temp <- aggregated_temp$X
  X_inside_humidity <- aggregated_humidity$X
  Y <- aggregated_temp$Y

  if (normalize_data) {
    X_inside_temp <- min_max_normalize(X_inside_temp)
    X_inside_humidity <- min_max_normalize(X_inside_humidity)
  }

  if (features == "temp") {
    X_combined <- X_inside_temp
  } else if (features == "humi") {
    X_combined <- X_inside_humidity
  } else if (features == "temp_humi") {
    X_combined <- cbind(X_inside_temp, X_inside_humidity)
  } else {
    stop("Unsupported feature set: ", features)
  }

  num_features <- get_number_of_features(features)
  X_combined <- as.matrix(X_combined)
  step_size <- time_steps - overlap
  number_of_samples <- floor((nrow(X_combined) - time_steps) / step_size) + 1

  if (number_of_samples <= 0) {
    stop("Not enough rows to create one full input window.")
  }

  X_array_all <- array(0, dim = c(number_of_samples, time_steps, num_features))
  y_vector <- numeric(number_of_samples)
  valid_count <- 0

  for (i in seq_len(number_of_samples)) {
    if (i %% 50000 == 0) {
      cat(i, "/", number_of_samples, " - ")
    }

    start_index <- ((i - 1) * step_size) + 1
    window_indices <- start_index:(start_index + time_steps - 1)
    sensor_numbers <- data$sensor.number[window_indices]

    # Avoid windows that accidentally cross from one room/sensor into another.
    if (length(unique(sensor_numbers)) != 1) {
      next
    }

    valid_count <- valid_count + 1
    window_matrix <- matrix(0, nrow = time_steps, ncol = num_features)
    for (j in seq_len(num_features)) {
      window_matrix[, j] <- X_combined[window_indices, j]
    }

    X_array_all[valid_count, , ] <- window_matrix
    # Keep the first label in the window to match the original implementation.
    y_vector[valid_count] <- Y[window_indices[1]]
  }

  if (valid_count == 0) {
    stop("No valid windows were created. Check sensor ordering and window size.")
  }

  X_array <- X_array_all[seq_len(valid_count), , , drop = FALSE]
  Y_array <- to_categorical_custom(y_vector[seq_len(valid_count)], num_classes = number_of_classes)

  cat("Dataset Shape: X:", dim(X_array), "Y:", dim(Y_array), "\n")

  if (type != "train") {
    return(X_array)
  }

  if (!include_test) {
    return(list(X_train = X_array, y_train = Y_array))
  }

  set.seed(123)
  label_int <- max.col(Y_array) - 1
  index_data <- data.frame(index = seq_len(dim(X_array)[1]), label = factor(label_int))

  split <- rsample::initial_split(index_data, prop = 0.85, strata = "label")
  train_val_index <- rsample::training(split)
  test_index <- rsample::testing(split)

  split_train_val <- rsample::initial_split(train_val_index, prop = 0.8235, strata = "label")
  train_index <- rsample::training(split_train_val)
  val_index <- rsample::testing(split_train_val)

  X_train <- X_array[train_index$index, , , drop = FALSE]
  y_train <- Y_array[train_index$index, , drop = FALSE]
  X_val <- X_array[val_index$index, , , drop = FALSE]
  y_val <- Y_array[val_index$index, , drop = FALSE]
  X_test <- X_array[test_index$index, , , drop = FALSE]
  y_test <- Y_array[test_index$index, , drop = FALSE]

  if (use_cache) {
    ensure_dir(folder_name)
    saveRDS(X_train, file.path(folder_name, "X_train.rds"))
    saveRDS(y_train, file.path(folder_name, "y_train.rds"))
    saveRDS(X_val, file.path(folder_name, "X_val.rds"))
    saveRDS(y_val, file.path(folder_name, "y_val.rds"))
    saveRDS(X_test, file.path(folder_name, "X_test.rds"))
    saveRDS(y_test, file.path(folder_name, "y_test.rds"))
  }

  list(
    X_train = X_train,
    y_train = y_train,
    X_val = X_val,
    y_val = y_val,
    X_test = X_test,
    y_test = y_test
  )
}

#' Store model test metrics and return the model file stem.
#'
#' The CSV is appended to `result/metrics_report_test.csv`. The returned file
#' name is used by `save_my_model()`.
#'
#' @param metrics Named vector/list returned by Keras `evaluate()`.
#' @param train_val_result Keras history object returned by `fit()`.
#' @param model_string Human-readable model identifier.
#' @param train_size Number of training windows.
#' @param val_size Number of validation windows.
#' @param test_size Number of test windows.
#' @param training_time Training duration in seconds.
#' @param epochs_num Number of epochs actually completed.
#' @param code Dataset code used in the output file name.
#' @param selected_feature Feature set used for training.
#' @param number_of_classes Number of target classes.
#' @param aggregate Aggregation identifier used in the output file name.
#' @param time_steps Window size.
#' @param overlap Window overlap.
#' @param sensor_data_rows Number of cleaned source rows.
#' @param model_details Extra text to append to the model file name.
#' @return Character file stem for saving the trained Keras model.
store_test_as_csv <- function(metrics,
                              train_val_result,
                              model_string,
                              train_size,
                              val_size,
                              test_size,
                              training_time,
                              epochs_num,
                              code,
                              selected_feature,
                              number_of_classes,
                              aggregate,
                              time_steps,
                              overlap,
                              sensor_data_rows,
                              model_details) {
  ensure_dir("result")

  model_name <- ifelse(
    stringr::str_detect(model_string, "GTN"),
    "GTN",
    ifelse(
      stringr::str_detect(model_string, "bid"),
      "Bidirectional_LSTM",
      ifelse(stringr::str_detect(model_string, "cnn"), "1D_CNN", "Unknown")
    )
  )

  sensor_modality <- ifelse(
    stringr::str_detect(selected_feature, "temp_humi"),
    "Temperature_Humidity",
    ifelse(
      stringr::str_detect(selected_feature, "humi"),
      "Humidity",
      ifelse(stringr::str_detect(selected_feature, "temp"), "Temperature", "Unknown")
    )
  )

  thresh <- 3
  metrics <- as.list(metrics)
  avg_epoch_time <- round(training_time / epochs_num, thresh)

  test_accuracy <- ifelse(metrics$accuracy < 1 / 10^thresh, 0, round(metrics$accuracy, thresh) * 100)
  test_recall <- ifelse(metrics$recall < 1 / 10^thresh, 0, round(metrics$recall, thresh) * 100)
  test_precision <- ifelse(metrics$precision < 1 / 10^thresh, 0, round(metrics$precision, thresh) * 100)
  test_loss <- ifelse(metrics$loss < 1 / 10^thresh, 0, round(metrics$loss, thresh))

  train_recall <- max(train_val_result$metrics$recall, na.rm = TRUE)
  train_precision <- max(train_val_result$metrics$precision, na.rm = TRUE)
  train_f1_score <- round(2 * (train_precision * train_recall) / (train_precision + train_recall), thresh) * 100

  val_recall <- max(train_val_result$metrics$val_recall, na.rm = TRUE)
  val_precision <- max(train_val_result$metrics$val_precision, na.rm = TRUE)
  val_f1_score <- round(2 * (val_precision * val_recall) / (val_precision + val_recall), thresh) * 100

  test_f1_score <- round(2 * (test_precision * test_recall) / (test_precision + test_recall), thresh)

  metrics_data <- data.frame(
    model = model_name,
    time_steps = time_steps,
    overlap = overlap,
    sensor_modality = sensor_modality,
    epochs = epochs_num,
    train_size = train_size,
    val_size = val_size,
    test_size = test_size,
    avg_epoch_time = avg_epoch_time,
    test_accuracy = test_accuracy,
    test_loss = test_loss,
    train_f1_score = train_f1_score,
    val_f1_score = val_f1_score,
    test_f1_score = test_f1_score
  )

  file_path <- "result/metrics_report_test.csv"
  write.table(
    metrics_data,
    file = file_path,
    sep = ",",
    col.names = !file.exists(file_path),
    row.names = FALSE,
    append = file.exists(file_path)
  )

  paste(
    code,
    "_Ep", epochs_num,
    "_Db", sensor_data_rows,
    "_Ts", time_steps,
    "_Ol", overlap,
    "_Ft", selected_feature,
    "_Cl", number_of_classes,
    "_Ag", aggregate,
    model_details,
    sep = ""
  )
}

#' Save a trained Keras model to disk.
#'
#' @param file_name File stem, without extension.
#' @param model Keras model object.
#' @param directory Output directory.
#' @return Invisibly returns the saved model path.
save_my_model <- function(file_name, model, directory = "model") {
  ensure_dir(directory)
  file_path <- file.path(directory, paste0(file_name, ".keras"))
  keras3::save_model(model, file_path, overwrite = TRUE)
  invisible(file_path)
}

#' Load a saved Keras model.
#'
#' @param directory Directory containing the model file.
#' @param model_name Model file name, including extension.
#' @return Loaded Keras model object.
load_my_model <- function(directory, model_name) {
  keras3::load_model(file.path(directory, model_name))
}

#' Predict labels for a windowed input array.
#'
#' @param model Trained Keras model.
#' @param test_data Original unwindowed data frame used only to align timestamps.
#' @param X_array Input array of shape `[samples, time_steps, features]`.
#' @param time_steps Window size.
#' @param overlap Window overlap.
#' @return Data frame containing the window inputs and predicted class labels.
predict_my_model <- function(model, test_data, X_array, time_steps, overlap) {
  prediction <- predict(model, X_array)

  if (any(is.na(prediction))) {
    prediction[is.na(prediction)] <- 0
  }

  prediction <- apply(prediction, 1, which.max) - 1
  prediction <- matrix(prediction, ncol = 1)

  X_df <- as.data.frame(X_array)
  batch_size <- time_steps - overlap
  batch_start_indices <- seq(1, nrow(test_data), by = batch_size)
  adjusted_timestamp <- test_data$X.timestamp[batch_start_indices][1:nrow(X_df)]
  adjusted_timestamp <- format(as.POSIXct(adjusted_timestamp, tz = "UTC"), "%Y-%b-%d %H:%M:%S")

  X_df %>%
    dplyr::mutate(timestamp = adjusted_timestamp) %>%
    dplyr::mutate(prediction = prediction) %>%
    dplyr::select(timestamp, dplyr::everything())
}

#' Expand window-level predictions back to the original time resolution.
#'
#' @param prediction Vector or one-column matrix of window-level predictions.
#' @param time_steps Window size.
#' @param overlap Window overlap.
#' @return Array with each prediction repeated by the model stride.
expand_prediction <- function(prediction, time_steps, overlap) {
  expand_times <- time_steps - overlap
  matrix_df <- as.data.frame(prediction)
  expanded_data <- matrix_df %>% tidyr::uncount(weights = expand_times)
  as.array(as.matrix(expanded_data))
}

#' Build the GTN-style attention model used in the AC experiments.
#'
#' @param time_steps Number of observations in each input window.
#' @param number_of_features Number of input feature channels.
#' @param number_of_classes Number of output classes.
#' @param model_activation Output activation, usually `"sigmoid"` or `"softmax"`.
#' @return Uncompiled Keras model.
create_GTN <- function(time_steps, number_of_features, number_of_classes, model_activation) {
  input_shape <- c(time_steps, number_of_features)
  input_data <- keras3::layer_input(shape = input_shape)

  channelwise <- input_data %>% keras3::layer_dense(units = 16)
  channelwise_attention <- keras3::layer_multi_head_attention(
    num_heads = 8,
    key_dim = 16
  )(
    query = channelwise,
    key = channelwise,
    value = channelwise
  ) %>%
    keras3::layer_add() %>%
    keras3::layer_normalization() %>%
    keras3::layer_dense(units = 64, activation = "relu") %>%
    keras3::layer_add() %>%
    keras3::layer_normalization()

  stepwise <- input_data %>% keras3::layer_dense(units = 16)
  stepwise_attention <- keras3::layer_multi_head_attention(
    num_heads = 8,
    key_dim = 16
  )(
    query = stepwise,
    key = stepwise,
    value = stepwise
  ) %>%
    keras3::layer_add() %>%
    keras3::layer_normalization() %>%
    keras3::layer_dense(units = 64, activation = "relu") %>%
    keras3::layer_add() %>%
    keras3::layer_normalization()

  combined <- keras3::layer_concatenate(list(channelwise_attention, stepwise_attention)) %>%
    keras3::layer_dense(units = 128, activation = "relu")

  output <- combined %>%
    keras3::layer_flatten() %>%
    keras3::layer_dense(units = number_of_classes) %>%
    keras3::layer_activation(model_activation)

  keras3::keras_model(inputs = input_data, outputs = output)
}

#' Summarise prediction errors over daily, weekly, monthly, or quarterly periods.
#'
#' @param inside_data Data frame with `X.timestamp`, `label`, `prediction`, and
#'   `sensor.temperature`.
#' @param period Aggregation period.
#' @return Aggregated error summary by period.
calculate_aggregates <- function(inside_data,
                                 period = c("weekly", "daily", "monthly", "quarterly")) {
  period <- match.arg(period)
  inside_data$label <- ifelse(inside_data$label != 0, 1, 0)
  inside_data$prediction <- ifelse(inside_data$prediction != 0, 1, 0)
  inside_data$X.timestamp <- as.POSIXct(
    inside_data$X.timestamp,
    format = "%Y-%m-%d %H:%M:%S",
    tz = "UTC"
  )

  floor_unit <- switch(
    period,
    weekly = "week",
    daily = "day",
    monthly = "month",
    quarterly = "quarter"
  )

  inside_data %>%
    dplyr::mutate(period = lubridate::floor_date(X.timestamp, floor_unit)) %>%
    dplyr::group_by(period) %>%
    dplyr::summarise(
      avg_inside_temperature = round(mean(sensor.temperature, na.rm = TRUE), 2),
      total_actual = round(sum(label != 0, na.rm = TRUE)),
      total_ac_on_minutes = round(sum(prediction != 0, na.rm = TRUE)),
      dif = total_ac_on_minutes - total_actual,
      error_percentage = (total_ac_on_minutes - total_actual) / total_actual * 100,
      total_mismatches = sum(prediction != label, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::arrange(period)
}

#' Calculate binary classification metrics for AC on/off prediction.
#'
#' @param labels Ground-truth labels. Non-zero values are treated as AC on.
#' @param prediction Predicted labels. Non-zero values are treated as AC on.
#' @return Named list containing MAE, MSE, RMSE, R-squared, F1 score, and accuracy.
calculate_metrics <- function(labels, prediction) {
  labels <- labels[seq_along(prediction)]
  labels <- ifelse(labels != 0, 1, 0)
  prediction <- ifelse(prediction != 0, 1, 0)

  dev <- labels - prediction
  mae <- mean(abs(dev), na.rm = TRUE) * 100
  mse <- mean(dev^2, na.rm = TRUE)
  rmse <- sqrt(mse)

  sse <- sum(dev^2, na.rm = TRUE)
  sst <- sum((labels - mean(labels, na.rm = TRUE))^2, na.rm = TRUE)
  r_squared <- ifelse(sst == 0, NA, 1 - (sse / sst))

  tp <- sum(labels == 1 & prediction == 1, na.rm = TRUE)
  tn <- sum(labels == 0 & prediction == 0, na.rm = TRUE)
  fp <- sum(labels == 0 & prediction == 1, na.rm = TRUE)
  fn <- sum(labels == 1 & prediction == 0, na.rm = TRUE)

  precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  f1_score <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall)) * 100
  accuracy <- (tp + tn) / length(labels) * 100

  list(
    MAE = mae,
    MSE = mse,
    RMSE = rmse,
    R_squared = r_squared,
    F1_Score = f1_score,
    Accuracy = accuracy
  )
}
