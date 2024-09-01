"""Compute the test perplexity of CAREER or baselines on the survey data."""
import argparse
import os
import pdb
import torch
import torch.nn.functional as F
from fairseq import utils

from fairseq.models.transformer import TransformerModel
from fairseq.models.bag_of_jobs import BagOfJobsModel
from fairseq.models.lstm import LSTMModel
from fairseq.models.regression import RegressionModel
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--binary-data-dir", 
                    type=str,
                    help="Location of binarized data.")
parser.add_argument("--save-dir", 
                    type=str,
                    help="Location of saved model checkpoints.")
parser.add_argument("--model-name", 
                    type=str,
                    default='career',
                    help="Model name (must be one of: 'career', 'bag-of-jobs',"
                         " or 'regression').")
parser.add_argument("--dataset-name", 
                    type=str,
                    help="Name of survey dataset (e.g. 'nlsy').")
args = parser.parse_args()

model_path = os.path.join(
  args.save_dir, 
  '{}/{}{}'.format(
    args.dataset_name, args.model_name, 
    "-transferred" if args.model_name == 'career' else ''))
binary_data_path = os.path.join(args.binary_data_dir, args.dataset_name)

# Load model.
if args.model_name == 'career':
  model = TransformerModel.from_pretrained(
    model_path,
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path=binary_data_path)
elif args.model_name == 'bag-of-jobs':
  model = BagOfJobsModel.from_pretrained(
    model_path,
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path=binary_data_path)
elif args.model_name == 'regression':
  model = RegressionModel.from_pretrained(
    model_path,
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path=binary_data_path)
else:
  raise ValueError("Model name must be one of: 'career', 'bag-of-jobs',"
                   " or 'regression'.")

# Move model to GPU and set to eval mode.
if torch.cuda.is_available():
  model.cuda()
model.eval()
model.model = model.models[0]
two_stage = model.model.decoder.args.two_stage

# Load test data.
model.task.load_dataset('test')
itr = model.task.get_batch_iterator(
  dataset=model.task.dataset('test'),
  max_tokens=1200,
  max_sentences=100,).next_epoch_itr(shuffle=False)

summed_nll = 0
total_tokens = 0
all_preds = {}  # full distribution over all possible tokens.
all_nlls = {}  # the NLL for the target token.

# keep track of data sizes to ensure completeness.
num_individuals = 0
num_observations = 0

for eval_index, sample in enumerate(itr):
  print("Evaluating {}/{}...".format(eval_index, itr.total))
  with torch.no_grad():
    if torch.cuda.is_available():
      sample = utils.move_to_cuda(sample)
    
    # Compute log probs for the sample.
    sample['net_input']['prev_output_tokens'] = sample[
      'net_input']['src_tokens']
    del sample['net_input']['src_tokens']
    output = model.model.decoder(**sample['net_input'])
    lprobs = model.model.get_normalized_probs(
      output, log_probs=True, two_stage=two_stage, 
      prev_tokens=sample['net_input']['prev_output_tokens'])
    batch_size, seq_len, _ = lprobs.size()
    target = sample['target'].to(model.device)
    # Make sure we don't evaluate <eos>.
    target[target == model.task.dictionary.eos_index] = (
      model.task.dictionary.pad_index)
    nll_loss = F.nll_loss(
      lprobs.transpose(2, 1), sample['target'].to(model.device), 
      ignore_index=model.model.decoder.padding_idx, reduction="none",)
    summed_nll += nll_loss.sum().item()
    total_tokens += target.ne(model.task.dictionary.pad_index).sum().item()
    # save the predicted distribution.
    # you where using prev_output.shape[0] in your code, I used batch_size instead. They are the same.
    prev_output = sample['net_input']['prev_output_tokens'].cpu().numpy()
    assert prev_output.shape[0] == batch_size
    probs = lprobs.exp()
    num_individuals += batch_size
    for batch_ind in range(batch_size):
      labels = sample['target'][batch_ind].cpu().numpy()
      mask = labels!= model.task.dictionary.pad_index  # only keep for non-padding labels.
      all_preds[sample['id'][batch_ind].item()] = probs[batch_ind].cpu().numpy()[mask]
      all_nlls[sample['id'][batch_ind].item()] = nll_loss[batch_ind].cpu().numpy()[mask]
      num_observations += len(labels[labels != 1])

print("..................")
print(f"Number of individuals: {num_individuals:,} and {num_observations:,} observations.")

overall_perplexity = np.exp(summed_nll / total_tokens)
print("..................")
print("Test-set results for {}, model loaded from '{}'".format(
  args.model_name, model_path))
print("..................")

print("Overall perplexity: {:.2f}".format(overall_perplexity))
print("..................")
# Save all_preds in to nll_{dataset}_{seed}.npy
# TODO: how to get several seeds?
seed = 1
output_path = "/oak/stanford/groups/athey/career_transformer_data/CAREERv1_results"
np.save(f"{output_path}/probs_{args.dataset_name}_{seed}.npy", all_preds)
np.save(f"{output_path}/nll_{args.dataset_name}_{seed}.npy", all_nlls)
# np.save(f"predictions/dict.txt", model.task.target_dictionary.symbols)
# all_preds is a dict, where each key is an integer, and the value is a numpy array of nlls.
# np.load("nll_{}_{}.npy".format(dataset, seed), allow_pickle=True).item()
