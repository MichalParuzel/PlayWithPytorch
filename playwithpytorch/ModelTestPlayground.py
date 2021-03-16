import torch
import torch.nn as nn

class ValidationTester():
    def train_model(self, model, dataloader, device, mapping, print_fail=False):
        model.eval()

        total = 0
        correct = 0
        wrong = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                m = nn.Softmax(dim=1)
                outputs = model(inputs)
                class_prob = m(outputs) *100
                _, preds = torch.max(outputs, 1)

                total += 1
                if preds.item() != labels.item():
                    wrong += 1
                    if print_fail:
                        predicted = mapping[preds.item()]
                        predicted_perc = round(class_prob.data[0, preds.item()].item(), 2)
                        should_be_predicted = mapping[labels.item()]
                        should_be_predicted_perc = round(class_prob.data[0, labels.item()].item(), 2)
                        print("Predicted {0} with {1} %".format(predicted, predicted_perc))
                        print("Should be {0} with {1} %".format(should_be_predicted, should_be_predicted_perc))
                else:
                    correct += 1

            if total % 100 == 0:
                print("total: {0}, correct: {1} wrong {2}".format(total, correct, wrong))
                print("Current acc: {0}%".format(correct/total * 100))

        print()

        """
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        """