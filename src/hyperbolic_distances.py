import time
from scipy.spatial.distance import squareform, pdist
import numpy as np
import argparse
import pickle

def hyperbolize(x):
  x = x.detach().numpy()
  n = [pdist(x[i], "sqeuclidean") for i in range(len(x))]
  MACHINE_EPSILON = np.finfo(np.double).eps
  m = [squareform(n[i]) for i in range(len(n))]
  m = np.stack(m, axis=0)
  qsqr = np.sum(x ** 2, axis=2)
  divisor1 = np.maximum(1 - qsqr[:, :, np.newaxis], initial=MACHINE_EPSILON)
  divisor2 = np.maximum(1 - qsqr[:, np.newaxis, :], initial=MACHINE_EPSILON)
  divisor = divisor1 * divisor2
  m = np.arccosh(1 + 2 * m / divisor ) #** 2
  return m

def main(opt):
  dataset = opt['dataset']
  for emb_dim in [16, 8, 4, 2]:
    with open(f"../data/pos_encodings/{dataset}_HYPS{emb_dim:02d}.pkl", "rb") as f:
      emb = pickle.load(f)
    t = time.time()
    sqdist = pdist(emb.detach().numpy(), "sqeuclidean")
    distances_ = hyperbolize(emb.detach().numpy(), sqdist)
    print("Distances calculated in %.2f sec" % (time.time()-t))
    #with open(f"../data/pos_encodings/{dataset}_HYPS{emb_dim:02d}_dists.pkl", "wb") as f:
    #  pickle.dump(distances, f)
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='ALL',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
  args = parser.parse_args()
  opt = vars(args)
  main(opt)