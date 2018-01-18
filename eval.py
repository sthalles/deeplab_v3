import metrics
import numpy as np

def compute_metrics(batch_predictions, batch_labels, class_labels):
    """

    :param batch_predictions: Prediction numpy array with shape [BATCH,HEIGHT,WIDTH]
    :param batch_labels: Labels numpy array with shape [BATCH,HEIGHT,WIDTH]
    :param class_labels: list containing valid classes id
    :return:
    """

    global_mean_IoU_list = []
    global_freq_IoU_list = []
    global_mean_acc_list = []
    global_pixel_acc_list = []

    num_of_examples = batch_predictions.shape[0]

    for class_id in class_labels:

        mean_IoU_list = []
        freq_IoU_list = []
        mean_acc_list = []
        pixel_acc_list = []

        # loop though all predicted images and ground thruths
        for i in range(num_of_examples):
            prediction = batch_predictions[i]
            ground_truth = batch_labels[i]

            # prediction = np.load("/home/thalles_silva/DataPublic/SemanticSegmentation/Projects/ssai-cnn/results/test_results_saito_pre_trained/" + image_name.strip() + "_pred_argmax.npy")
            #prediction = np.squeeze(prediction)
            #ground_truth = np.squeeze(ground_truth)

            prediction_binary = np.zeros(prediction.shape, dtype=np.uint8)
            label_binary = np.zeros(prediction.shape, dtype=np.uint8)

            prediction_binary[np.where( prediction ==class_id)] = 1
            label_binary[np.where( ground_truth ==class_id)] = 1

            IoU = metrics.mean_IU(label_binary, prediction_binary)
            freq_IoU = metrics.frequency_weighted_IU(prediction_binary, label_binary)
            mean_acc = metrics.mean_accuracy(prediction_binary, label_binary)
            pixel_acc = metrics.pixel_accuracy(prediction_binary, label_binary)

            mean_IoU_list.append(IoU)
            freq_IoU_list.append(freq_IoU)
            mean_acc_list.append(mean_acc)
            pixel_acc_list.append(pixel_acc)

        global_mean_IoU_list.append(mean_IoU_list)
        global_freq_IoU_list.append(freq_IoU_list)
        global_mean_acc_list.append(mean_acc_list)
        global_pixel_acc_list.append(pixel_acc_list)

    return np.mean(global_mean_IoU_list), np.mean(global_freq_IoU_list), np.mean(global_mean_acc_list), np.mean(global_pixel_acc_list)