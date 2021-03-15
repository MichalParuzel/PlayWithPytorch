import torch
import torch.nn as nn

class ValidationTester():
    def train_model(self, model, dataloader, device):
        model.eval()

        # Iterate over data.
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                m = nn.Softmax(dim=1)
                outputs = model(inputs)
                class_prob = m(outputs) *100

                _, preds = torch.max(outputs, 1)
        print()

        """
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        """