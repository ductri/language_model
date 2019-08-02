import time
import logging

import numpy as np
import torch

from model_def.T_LM import constants


def train(training_model, train_loader, eval_loader, device, num_epoch=10, print_every=1000, predict_every=500,
          eval_every=500, input_transform=None, output_transform=None, init_step=0, training_checker=None, my_logger=None,
          bos_id=None, eos_id=None):
    if input_transform is None:
        input_transform = lambda *x: x
    if output_transform is None:
        output_transform = lambda *x: x
    model = training_model.model

    def predict_and_print_sample(inputs):
        sample_size = 5
        input_tensors = [input_tensor[:sample_size] for input_tensor in inputs]
        predict_tensor = model(input_tensors[0], max_length=constants.MAX_LEN, bos_id=2, eos_id=3)

        input_transformed = input_transform(input_tensors[0].cpu().numpy(), input_tensors[1].cpu().numpy())
        predict_transformed = output_transform(predict_tensor.cpu().numpy(), input_tensors[1].cpu().numpy())

        for idx, (src, pred) in enumerate(zip(input_transformed, predict_transformed)):
            logging.info('Sample %s ', idx + 1)
            logging.info('Source:\t%s', src)
            logging.info('Predict:\t%s', pred)
            logging.info('------')

    t_loss_tracking = []
    t_ppl_tracking = []

    t_loss_mean_tag = 'train/loss_mean'
    t_loss_std_tag = 'train/loss_std'
    t_ppl_mean_tag = 'train/ppl_mean'
    t_ppl_std_tag = 'train/ppl_std'
    e_loss_mean_tag = 'eval/loss_mean'
    e_loss_std_tag = 'eval/loss_std'
    e_ppl_mean_tag = 'eval/ppl_mean'
    e_ppl_std_tag = 'eval/ppl_std'
    lr_tag = 'lr'
    train_duration_tag = 'train/step_duration'
    eval_duration_tag = 'eval/step_duration'

    step = init_step
    training_model.to(device)

    logging.info('----------------------- START TRAINING -----------------------')
    for _ in range(num_epoch):
        for inputs in train_loader:
            inputs = [i.to(device) for i in inputs]
            start = time.time()
            t_loss = training_model.train_batch(inputs[0], inputs[1])
            t_loss_tracking.append(t_loss)
            t_ppl_tracking.append(np.exp(t_loss))
            step += 1
            with torch.no_grad():
                if step % print_every == 0 or step == 1:
                    model.eval()
                    my_logger.add_scalar(t_loss_mean_tag, np.mean(t_loss_tracking), step)
                    my_logger.add_scalar(t_loss_std_tag, np.std(t_loss_tracking), step)
                    my_logger.add_scalar(t_ppl_mean_tag, np.mean(t_ppl_tracking), step)
                    my_logger.add_scalar(t_ppl_std_tag, np.std(t_ppl_tracking), step)
                    my_logger.add_scalar(lr_tag, training_model.get_lr(), step)
                    my_logger.add_scalar(train_duration_tag, time.time() - start, step)
                    t_loss_tracking.clear()
                    t_ppl_tracking.clear()

                if step % predict_every == 0:
                    model.eval()

                    logging.info('\n\n------------------ Predict samples from train ------------------ ')
                    logging.info('Step: %s', step)
                    predict_and_print_sample(inputs)

                if step % eval_every == 0:
                    model.eval()
                    start = time.time()
                    e_loss_tracking = []
                    e_ppl_tracking = []
                    for eval_inputs in eval_loader:
                        eval_inputs = [i.to(device) for i in eval_inputs]
                        e_loss = training_model.get_loss(eval_inputs[0], eval_inputs[1])
                        e_loss = e_loss.cpu().item()
                        e_loss_tracking.append(e_loss)
                        e_ppl_tracking.append(np.exp(e_loss))

                    logging.info('\n\n------------------ \tEvaluation\t------------------')
                    logging.info('Number of batchs: %s', len(e_loss_tracking))
                    my_logger.add_scalar(e_loss_mean_tag, np.mean(e_loss_tracking), step)
                    my_logger.add_scalar(e_loss_std_tag, np.std(e_loss_tracking), step)
                    my_logger.add_scalar(e_ppl_mean_tag, np.mean(e_ppl_tracking), step)
                    my_logger.add_scalar(e_ppl_std_tag, np.std(e_ppl_tracking), step)
                    my_logger.add_scalar(eval_duration_tag, time.time()-start, step)

                    training_checker.update(-np.mean(e_loss_tracking), step)
                    best_score, best_score_step = training_checker.best()
                    logging.info('Current best score: %s recorded at step %s', best_score, best_score_step)

                    eval_inputs = next(iter(eval_loader))
                    eval_inputs = [item.to(device) for item in eval_inputs]
                    predict_and_print_sample(eval_inputs)


