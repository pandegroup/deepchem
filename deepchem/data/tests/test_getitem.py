import deepchem as dc
import numpy as np
import os


def test_numpy_getindex_int():
  """Test __getitem__ on int for NumpyDataset"""
  X = np.random.rand(10, 10)
  y = np.random.rand(10,)
  w = np.random.rand(10,)
  dataset = dc.data.NumpyDataset(X, y, w)
  xi, yi, wi, idsi = dataset[5]
  assert np.all(xi == X[5])
  assert np.all(yi == y[5])
  assert np.all(wi == w[5])
  assert idsi == 5


def test_numpy_getindex_slice():
  """Test __getitem__ on int for NumpyDataset"""
  X = np.random.rand(10, 10)
  y = np.random.rand(10,)
  w = np.random.rand(10,)
  dataset = dc.data.NumpyDataset(X, y, w)
  start = 3
  for (xi, yi, wi, idsi) in dataset[3:5]:
    assert np.all(xi == X[start])
    assert np.all(yi == y[start])
    assert np.all(wi == w[start])
    assert idsi == start
    start += 1


def test_disk_getindex_int():
  """Test __getitem__ on int for DiskDataset"""
  X = np.random.rand(10, 10)
  y = np.random.rand(10,)
  w = np.random.rand(10,)
  dataset = dc.data.DiskDataset.from_numpy(X, y, w)
  xi, yi, wi, idsi = dataset[5]
  assert np.all(xi == X[5])
  assert np.all(yi == y[5])
  assert np.all(wi == w[5])
  assert idsi == 5


def test_disk_getindex_slice():
  """Test __getitem__ on slice for DiskDataset"""
  X = np.random.rand(10, 10)
  y = np.random.rand(10,)
  w = np.random.rand(10,)
  dataset = dc.data.DiskDataset.from_numpy(X, y, w)
  start = 3
  for (xi, yi, wi, idsi) in dataset[3:5]:
    assert np.all(xi == X[start])
    assert np.all(yi == y[start])
    assert np.all(wi == w[start])
    assert idsi == start
    start += 1


def test_image_getindex_int():
  """Test __getitem__ on int for ImageDataset"""
  path = os.path.join(os.path.dirname(__file__), 'images')
  files = [os.path.join(path, f) for f in os.listdir(path)]
  ds = dc.data.ImageDataset(files, np.random.random(10))
  xi, yi, wi, idsi = ds[5]
  assert np.all(xi == ds.X[5])
  assert np.all(yi == ds.y[5])
  assert np.all(wi == ds.w[5])
  assert np.all(idsi == ds.ids[5])


def test_image_getindex_slice():
  """Test __getitem__ on slice for ImageDataset"""
  path = os.path.join(os.path.dirname(__file__), 'images')
  files = [os.path.join(path, f) for f in os.listdir(path)]
  ds = dc.data.ImageDataset(files, np.random.random(10))
  start = 3
  for (xi, yi, wi, idsi) in ds[3:5]:
    assert np.all(xi == ds.X[start])
    assert np.all(yi == ds.y[start])
    assert np.all(wi == ds.w[start])
    assert idsi == ds.ids[start]
    start += 1
