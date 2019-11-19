import time
import torch
from schemes.model_metrics import AverageMeter
from schemes.model_metrics import calculate_accuracy_percent
from utils.trivial_definition import separator_line
import sklearn
from utils.plot_utilities import generate_confusion_matrix


def validate_model(args, logger, val_loader,
                   model, metrics, criterion,
                   epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    samples_count = AverageMeter()
    samples_right = AverageMeter()


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        y_true = []
        y_pred = []
        for i, (input, target, _) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # input1 = input[0]
            # input2 = input[1]
            if args.cuda is not None:
                # input1 = input1.cuda(non_blocking=True)
                # input1 = input1.float()
                # input2 = input2.cuda(non_blocking=True)
                # input2 = input2.float()
                input = input.cuda(non_blocking=True)

                target = target.cuda(non_blocking=True)
                input = input.float()
            # compute output

            # input1 = input1.permute(0, 2, 1, 3, 4).contiguous()
            # input2 = input2.permute(0, 2, 1, 3, 4).contiguous()
            input = input.permute(0, 2, 1, 3, 4).contiguous()

            output = model(input)
            loss = criterion(output, target)

            y_true.extend(target.tolist())
            _, pred = output.topk(1, 1, True, True)
            y_pred.extend(pred.t().tolist()[0])

            # measure accuracy and record loss
            if "accuracy_percent" in metrics:
                predicted_accuracy, n_correct_elems = calculate_accuracy_percent(output, target)
                samples_count.update(input.size(0))
                samples_right.update(n_correct_elems.item())
                accuracy.update(predicted_accuracy.item(), input.size(0))

            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            """
            if i % args.log_interval == 0:
                print('Test: [{0}/{1}]--'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})--'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})--'
                      'Prec@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       acc=accuracy))
            """

        if args.plot_confusion_matrix:
            class_names = val_loader.class_names
            plt_fig = generate_confusion_matrix(y_true, y_pred, class_names)
            args.tb_writer.add_figure('confusion matrix', plt_fig, epoch)

        logger.info('=> Validate:  '
                    'Elapse: {data_time.sum:.2f}/{sum_time.sum:.2f}s  '
                    'Loss: {loss.avg:.4f}  '
                    'Accuracy: {acc.avg:.2f}% '
                    '[{right:.0f}/{count:.0f}]'.format(loss=losses,
                                                  data_time=data_time,
                                                  sum_time=batch_time,
                                                  acc=accuracy,
                                                  right=samples_right.sum,
                                                  count=samples_count.sum))

        # recall
        if "precision_recall_percent" in metrics:
            precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro")
            recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")

            precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")
            recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")

            logger.info('Precision_micro  {:.4f}  '
                        'Recall_micro  {:.4f}  '
                        'Precision_weighted  {:.4f}  '
                        'Recall_weighted  {:.4f}  '
                        .format(precision_micro,
                                recall_micro,
                                precision_weighted,
                                recall_weighted))

            check_pred = precision_weighted

        else:
            check_pred = accuracy.avg

        logger.info(separator_line())

    return check_pred, losses.avg
