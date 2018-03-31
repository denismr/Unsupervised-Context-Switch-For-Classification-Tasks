import argparse
import numpy as np
import pandas as pd
import random
import math

from ForgettingBuffer import ForgettingBuffer as FB
from LeaveOneOut import LeaveOneOut as LOO
from IKSSW import IKSSW

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()

parser.add_argument('--title', dest='title', default='')
parser.add_argument('-exp', '--experiment', type=str, default='')
parser.add_argument('-cl', '--classifier', type=str, default='SVC')
parser.add_argument('-it', '--iterations', type=int, default=1)

args = parser.parse_args()

CCons = {
  'SVC': (SVC, {'probability':True}),
  'RF': (RandomForestClassifier, {}),
  'MLP': (MLPClassifier, {}),
}

def CreateClassifier():
  constructor, kw = CCons[args.classifier]
  return constructor(**kw)

experiments = {
  'WBFInsects': {
    'window_size': 100,
    'context_length': 1000,
    'dataset': 'WBFInsects',
    'ctx_feature': 'context',
    'ctx_feature_values': [1, 2],
    'target': 'species',
    'features': ['wbf'],
    'test_on': 'wbf',
  },
  'AedesQuinx': {
    'window_size': 100,
    'context_length': 900,
    'dataset': 'AedesQuinx',
    'ctx_feature': 'temp_range',
    'ctx_feature_values': [1, 2, 3, 4, 5, 6],
    'target': 'species',
    'features': ["wbf","eh_1","eh_2","eh_3","eh_4","eh_5","eh_6","eh_7","eh_8","eh_9","eh_10","eh_11","eh_12","eh_13","eh_14","eh_15","eh_16","eh_17","eh_18","eh_19","eh_20","eh_21","eh_22","eh_23","eh_24","eh_25"],
    'test_on': 'wbf',
  },
  'AedesSex': {
    'window_size': 100,
    'context_length': 900,
    'dataset': 'AedesSex',
    'ctx_feature': 'temp_range',
    'ctx_feature_values': [1, 2, 3, 4, 5, 6],
    'target': 'sex',
    'features': ["wbf","eh_1","eh_2","eh_3","eh_4","eh_5","eh_6","eh_7","eh_8","eh_9","eh_10","eh_11","eh_12","eh_13","eh_14","eh_15","eh_16","eh_17","eh_18","eh_19","eh_20","eh_21","eh_22","eh_23","eh_24","eh_25"],
    'test_on': 'wbf',
  },
  'ArabicDigit': {
    'window_size': 150,
    'context_length': 800,
    'dataset': 'ArabicDigit',
    'ctx_feature': 'sex',
    'ctx_feature_values': ['male', 'female'],
    'target': 'digit',
    'features': lambda x:  x != 'sex' and x != 'digit',
    'test_on': 'mfcc_1_mu',
  },
  'ArabicSex': {
    'window_size': 50,
    'context_length': 800,
    'dataset': 'ArabicDigit',
    'ctx_feature': 'digit',
    'ctx_feature_values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'target': 'sex',
    'features': lambda x:  x != 'sex' and x != 'digit',
    'test_on': 'mfcc_1_mu',
  },
  'HandwrittenLetter': {
    'window_size': 50,
    'context_length': 250,
    'dataset': 'Handwritten',
    'ctx_feature': 'author',
    'ctx_feature_values': [
      'Andre', 'Antonio', 'Denis', 'Diego', 'Felipe',
      'Gustavo', 'Minatel', 'Rita', 'Roberta', 'Sanches',
    ],
    'target': 'letter',
    'features': lambda x:  x != 'author' and x != 'letter',
    'test_on': 'glob_area',
  },
  'HandwrittenAuthor': {
    'window_size': 50,
    'context_length': 500,
    'dataset': 'Handwritten',
    'ctx_feature': 'letter',
    'ctx_feature_values': ['g', 'q', 'p'],
    'target': 'author',
    'features': lambda x:  x != 'author' and x != 'letter',
    'test_on': 'glob_area',
  },
}

types_detections = [
  '          KS on Cl Output',
  '      Kuiper on Cl Output',
  '            KS on Feature',
  '        Kuiper on Feature',
]

if not args.experiment in experiments:
  print('Invalid experiment')
  exit(0)

exp = experiments[args.experiment]
dataset = exp['dataset']
ctx_feature = exp['ctx_feature']
ctx_feature_values = exp['ctx_feature_values']
target = exp['target']
context_length = exp['context_length']
window_size = exp['window_size']
test_on = exp['test_on']

number_of_contexts = 0
stream_size = 0

data = pd.read_csv('data/%s.csv' % dataset)

features = exp['features'] if isinstance(exp['features'], list) \
  else list(filter(exp['features'], data.columns))

Tdata = list(data[test_on])
fdata = data[features]
ydata = data[target]
labels = list(ydata)

def FitWithIndices(classifier, indices):
  X = fdata.iloc[indices]
  Y = ydata.iloc[indices]
  classifier.fit(X, Y)
  return classifier

def NewClassifier(train):
  return FitWithIndices(CreateClassifier(), train)

def Classify(classifier, x):
  X = fdata.iloc[[x]]
  label = classifier.predict(X)[0]
  value = max(classifier.predict_proba(X)[0])
  value = math.floor(value * 100000) + random.random()
  return label, value

def LOOT(train):
  tab = []
  for tr, te in LOO(train):
    classifier = CreateClassifier()
    FitWithIndices(classifier, tr)
    _, myval = Classify(classifier, te)
    tab.append(myval)
  return tab

def RevTab(tab, rev, tabval):
  for v in tab:
    rev[v] = tabval

def GetDataForClassifiers(by, sz):
  data_for = []
  for i, v in enumerate(by):
    data_for.append(v[:sz])
    by[i] = v[sz:]
  return data_for

def GetNumberOfContexts(by, ctx_len):
  return min(map(lambda x: len(x) // ctx_len, by))

def CreateStream(by_feature, number_of_contexts, clen):
  tabs = []
  for v in by_feature:
    for i in range(number_of_contexts):
      t = []
      for j in range(clen):
        t.append(v[-1])
        v.pop()
      tabs.append(t)
  random.shuffle(tabs)
  return sum(tabs, [])

tot_ctx_acc = [[], [], [], []]
tot_acc = [[], [], [], []]
tot_acc_top = []
tot_acc_base = []
tot_acc_rbase = []

for iteration in range(args.iterations):
  by_feature = []
  rev_feature = {}

  for i, feature_value in enumerate(ctx_feature_values):
    by_feature.append(list(data[data[ctx_feature] == feature_value].index))
    random.shuffle(by_feature[i])
    RevTab(by_feature[i], rev_feature, i)
  
  true_ctx = rev_feature
  data_for_classifiers = GetDataForClassifiers(by_feature, window_size)
  number_of_contexts = GetNumberOfContexts(by_feature, context_length)
  stream = CreateStream(by_feature, number_of_contexts, context_length)

  classifiers = list(map(lambda x: NewClassifier(x), data_for_classifiers))
  baseline = NewClassifier(sum(data_for_classifiers, []))

  iks = list(map(lambda x: IKSSW(LOOT(x)), data_for_classifiers))
  iksf = list(map(lambda x: IKSSW(list(map(lambda y: Tdata[y], x))), data_for_classifiers))

  sorts = [
    lambda x: iks[x].KS(),
    lambda x: iks[x].Kuiper(),
    lambda x: iksf[x].KS(),
    lambda x: iksf[x].Kuiper(),
  ]

  correct_ctxs = [0, 0, 0, 0]
  corrects = [0, 0, 0, 0]

  corrects_base = 0
  corrects_rbase = 0
  corrects_top = 0

  ctx = list(range(len(classifiers)))

  stream_size = len(stream)
  test = stream

  for v in test:
    correct_label = labels[v]
    correct_ctx = true_ctx[v]

    for x, c in zip(iks, classifiers):
      x(Classify(c, v)[1])

    for x, c in zip(iksf, classifiers):
      x(Tdata[v])

    for i, k in enumerate(sorts):
      ctx.sort(key=k)
      predicted_ctx = ctx[0]
      predicted_label = Classify(classifiers[predicted_ctx], v)[0]
      corrects[i] += 1 if predicted_label == correct_label else 0
      correct_ctxs[i] += 1 if predicted_ctx == correct_ctx else 0
    
    corrects_base += 1 if Classify(baseline, v)[0] == correct_label else 0
    corrects_rbase += 1 if Classify(classifiers[random.choice(ctx)], v)[0] == correct_label else 0
    corrects_top += 1 if Classify(classifiers[correct_ctx], v)[0] == correct_label else 0

  for i in range(len(sorts)):
    tot_acc[i].append(100 * corrects[i] / len(test))
    tot_ctx_acc[i].append(100 * correct_ctxs[i] / len(test))

  tot_acc_rbase.append(100 * corrects_rbase / len(test))
  tot_acc_base.append(100 * corrects_base / len(test))
  tot_acc_top.append(100 * corrects_top / len(test))

if len(args.title) > 0:
  empty = '###%s###' % ('#' * len(args.title))
  print(empty)
  print('#  %s  #' % (args.title))
  print(empty)

def Mean(v):
  mu = sum(v) / len(v)
  sdev = math.sqrt(sum(map(lambda x: (x - mu) ** 2, v))/(len(v) - 1)) if len(v) > 1 else 0
  return mu, sdev

print('Configuration')
print('----------------')
print('               Classifier  |  %s' % args.classifier)
print('                  Dataset  |  %s' % dataset)
print('                   Target  |  %s' % target)
print('          Context feature  |  %s' % ctx_feature)
print('               # contexts  |  %d' % len(ctx_feature_values))
print('              Window size  |  %d' % window_size)
print('         Test stream size  |  %d' % stream_size)

print('           Concept length  |  %d' % context_length)
print('       Concept recurrence  |  %d' % number_of_contexts)
print('             # iterations  |  %d' % args.iterations)


for i, v in enumerate(types_detections):
  print('----------------')
  print(v)
  print('        Context accuracy: %.2f%% (sdev: %.2f)' % Mean(tot_ctx_acc[i]))
  print(' Classification accuracy: %.2f%% (sdev: %.2f)' % Mean(tot_acc[i]))

print('----------------')
print(' Single Clas. Base. acc.: %.2f%% (sdev: %.2f)' % Mean(tot_acc_base))
print('  Rand. Clas. Base. acc.: %.2f%% (sdev: %.2f)' % Mean(tot_acc_rbase))
print('        Topline accuracy: %.2f%% (sdev: %.2f)' % Mean(tot_acc_top))
print('----------------')
print()