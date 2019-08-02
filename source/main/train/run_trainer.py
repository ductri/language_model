import logging

import torch
from naruto_skills.training_checker import TrainingChecker
from naruto_skills.dl_logging import DLLoggingHandler, get_logger_instance, DLTBHandler
from naruto_skills import pytorch_utils

from data_for_train import dataset as my_dataset
from model_def.T_LM.model import Model
from model_def.T_LM.model_training import ModelTraining
from train.trainer import train


def tensor2text(x, seq_len):
    """

    :param x: array numpy 2d
    :param seq_len:
    :return:
    """
    x = x.tolist()
    pred = [my_dataset.voc.decode_ids(doc) for doc in x]
    return pred


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    EXP_ID = '1.3'
    ROOT_DIR = '/source/main/train/output/'
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    NUM_WORKERS = 0
    PRINT_EVERY = 100
    PREDICT_EVERY = 5000
    EVAL_EVERY = 10000
    PRE_TRAINED_MODEL = ''

    my_dataset.bootstrap()
    train_loader, eval_loader = my_dataset.get_datasets_2(BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    model = Model(Model.get_word_embedding(), d_model=640, num_layers=10, num_heads=10, rate=0.1)
    model.to(device)
    logging.info('Total trainable parameters: %s', pytorch_utils.count_parameters(model))
    model_training = ModelTraining(model)

    training_checker = TrainingChecker(model_training, root_dir='%s/saved_models' % ROOT_DIR + '/%s/%s'
                                                                % (model.__class__.__name__, EXP_ID), init_score=-1e9)
    my_logger = get_logger_instance('root')
    my_logger.add_handler(DLLoggingHandler())
    my_logger.add_handler(DLTBHandler(ROOT_DIR + '/logging/%s/%s' % (model.__class__.__name__, EXP_ID)))

    init_step = 0
    # Restore model
    if PRE_TRAINED_MODEL != '':
        checkpoint = torch.load(PRE_TRAINED_MODEL, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        init_step = checkpoint.get('step')
        logging.info('Optimizer: %s', model.optimizer)
        logging.info('Inspect learning rate after loading weights', model.get_lr())

        logging.info('Load pre-trained model from %s successfully', PRE_TRAINED_MODEL)

    train(model_training, train_loader, eval_loader, device=device, num_epoch=NUM_EPOCHS, print_every=PRINT_EVERY, predict_every=PREDICT_EVERY, eval_every=EVAL_EVERY, input_transform=tensor2text, output_transform=tensor2text, init_step=init_step, my_logger=my_logger, training_checker=training_checker)
