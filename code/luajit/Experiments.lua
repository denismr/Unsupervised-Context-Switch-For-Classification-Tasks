math.randomseed(os.time())

require 'vendor.fun.fun'()
require 'support.BlockGlobal'
local IB1 = require 'support.IB1'
local CSV = require 'support.CSVReader'
local Shuffle = require 'support.Shuffle'
local Split = require 'support.Split'
local IKSSW = require 'support.IKSSW'
local MergeTables = require 'support.MergeTables'
local Mean = require 'support.Mean'

local flags = require 'support.flags' {
  ['exp'] = '',
  ['title'] = '',
  ['it'] = 1, -- number of iterations
}:processArgs(arg)

local experiments = {
  WBFInsects = {
    window_size = 100,
    context_length = 1000,
    dataset = 'WBFInsects',
    ctx_feature = 'context',
    ctx_feature_values = {1, 2},
    target = 'species',
    features = {'wbf'},
    test_on = 'wbf',
  },
  AedesQuinx = {
    window_size = 100,
    context_length = 900,
    dataset = 'AedesQuinx',
    ctx_feature = 'temp_range',
    ctx_feature_values = {1, 2, 3, 4, 5, 6},
    target = 'species',
    features = {"wbf","eh_1","eh_2","eh_3","eh_4","eh_5","eh_6","eh_7","eh_8","eh_9","eh_10","eh_11","eh_12","eh_13","eh_14","eh_15","eh_16","eh_17","eh_18","eh_19","eh_20","eh_21","eh_22","eh_23","eh_24","eh_25"},
    test_on = 'wbf',
  },
  AedesSex = {
    window_size = 100,
    context_length = 900,
    dataset = 'AedesSex',
    ctx_feature = 'temp_range',
    ctx_feature_values = {1, 2, 3, 4, 5, 6},
    target = 'sex',
    features = {"wbf","eh_1","eh_2","eh_3","eh_4","eh_5","eh_6","eh_7","eh_8","eh_9","eh_10","eh_11","eh_12","eh_13","eh_14","eh_15","eh_16","eh_17","eh_18","eh_19","eh_20","eh_21","eh_22","eh_23","eh_24","eh_25"},
    test_on = 'wbf',
  },
  ArabicDigit = {
    window_size = 150,
    context_length = 800,
    dataset = 'ArabicDigit',
    ctx_feature = 'sex',
    ctx_feature_values = {'male', 'female'},
    target = 'digit',
    features = function(x) return x ~= 'sex' and x ~= 'digit' end,
    test_on = 'mfcc_1_mu',
  },
  ArabicSex = {
    window_size = 50,
    context_length = 800,
    dataset = 'ArabicDigit',
    ctx_feature = 'digit',
    ctx_feature_values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    target = 'sex',
    features = function(x) return x ~= 'sex' and x ~= 'digit' end,
    test_on = 'mfcc_1_mu',
  },
  HandwrittenLetter = {
    window_size = 50,
    context_length = 250,
    dataset = 'Handwritten',
    ctx_feature = 'author',
    ctx_feature_values = {
      'Andre', 'Antonio', 'Denis', 'Diego', 'Felipe',
      'Gustavo', 'Minatel', 'Rita', 'Roberta', 'Sanches',
    },
    target = 'letter',
    features = function(x) return x ~= 'author' and x ~= 'letter' end,
    test_on = 'glob_area',
  },
  HandwrittenAuthor = {
    window_size = 50,
    context_length = 500,
    dataset = 'Handwritten',
    ctx_feature = 'letter',
    ctx_feature_values = {'g', 'q', 'p'},
    target = 'author',
    features = function(x) return x ~= 'author' and x ~= 'letter' end,
    test_on = 'glob_area',
  },
}

local types_detections = {
  '          KS on Cl Output',
  '      Kuiper on Cl Output',
  '            KS on Feature',
  '        Kuiper on Feature',
}

if not experiments[flags.exp] then
  print 'Invalid experiment'
  return
end

local exp = experiments[flags.exp]

local dataset = exp.dataset
local ctx_feature = exp.ctx_feature
local ctx_feature_values = exp.ctx_feature_values
local target = exp.target
local context_length = exp.context_length
local window_size = exp.window_size
local test_on = exp.test_on

local number_of_contexts
local stream_size

local full_data, full_header = CSV('data/' .. dataset .. '.csv')
local features = type(exp.features) == 'table' and exp.features
                  or filter(exp.features, full_header):totable()

local function PopNTo(array, N, to)
  for i = 1, N do
    to[#to + 1] = array[#array]
    array[#array] = nil
  end
end

local function RevTab(tab, rev, val)
  for i, v in ipairs(tab) do
    rev[v] = val
  end
end

local function FFeat(f, val) return function(x) return x[f] == val end end
local function Feat(f) return function(x) return x[f] end end

local function GetDataForClassifiers(by, sz)
  local data_for = {}
  for i, v in ipairs(by) do
    data_for[i], by[i] = Split(v, sz)
  end
  return data_for
end

local function GetNumberOfContexts(by, ctx_len)
  return min(map(function(x) return math.floor(#x / ctx_len) end, by))
end

local function CreateStream(by_feature, number_of_contexts, clen)
  local tabs = {}
  for _, v in ipairs(by_feature) do
    for i = 1, number_of_contexts do
      local t = {}
      PopNTo(v, clen, t)
      tabs[#tabs + 1] = t
    end
  end
  Shuffle(tabs)
  return MergeTables(tabs)
end

local tot_ctx_acc = {{}, {}, {}, {}}
local tot_acc = {{}, {}, {}, {}}
local tot_acc_top = {}
local tot_acc_base = {}
local tot_acc_rbase = {}

for it = 1, flags.it do
  Shuffle(full_data)

  local by_feature = {}
  local rev_feature = {}

  for i, feature_value in ipairs(ctx_feature_values) do
    by_feature[i] = filter(FFeat(ctx_feature, feature_value), full_data):totable()
    RevTab(by_feature[i], rev_feature, i)
  end

  local true_ctx = rev_feature
  local data_for_classifiers = GetDataForClassifiers(by_feature, window_size)
  number_of_contexts = GetNumberOfContexts(by_feature, context_length)
  local stream = CreateStream(by_feature, number_of_contexts, context_length)

  local classifiers = totable(
    map(function(x)
      return IB1 {
        data = x,
        features = features,
        target = target,
      }
    end, data_for_classifiers)
  )

  local baseline = IB1 {
    data = chain(unpack(data_for_classifiers)):totable(),
    features = features,
    target = target,
  }

  local iks = map(function(x) return IKSSW(x:LOOT(), true) end, classifiers):totable()
  local iksf = map(function(x) return IKSSW(totable(map(Feat(test_on), x)), true) end, data_for_classifiers):totable()

  local sorts = {
    function(a, b) return iks[a]:KS() < iks[b]:KS() end,
    function(a, b) return iks[a]:Kuiper() < iks[b]:Kuiper() end,
    function(a, b) return iksf[a]:KS() < iksf[b]:KS() end,
    function(a, b) return iksf[a]:Kuiper() < iksf[b]:Kuiper() end,
  }

  local correct_ctxs = {0, 0, 0, 0}
  local corrects = {0, 0, 0, 0}

  local corrects_base = 0
  local corrects_rbase = 0
  local corrects_top = 0

  local ctx = totable(range(#classifiers))

  stream_size = #stream
  local test = stream
  for i, v in ipairs(test) do
    local correct_label = v[target]
    local correct_ctx = true_ctx[v]
    
    each(function(x, c) x:Increment(select(2, c(v))) end, zip(iks, classifiers))
    each(function(x) x:Increment(v[test_on]) end, iksf)

    for i, f in ipairs(sorts) do
      table.sort(ctx, f)
      local predicted_ctx = ctx[1]
      local predicted_label = classifiers[predicted_ctx](v)
      corrects[i] = corrects[i] + (predicted_label == correct_label and 1 or 0)
      correct_ctxs[i] = correct_ctxs[i] + (predicted_ctx == correct_ctx and 1 or 0)
    end
    
    corrects_base = corrects_base + (baseline(v) == correct_label and 1 or 0)
    corrects_rbase = corrects_rbase + (classifiers[math.random(#classifiers)](v) == correct_label and 1 or 0)
    corrects_top = corrects_top + (classifiers[correct_ctx](v) == correct_label and 1 or 0)
  end

  for i = 1, #sorts do
    table.insert(tot_acc[i], 100 * corrects[i] / #test)
    table.insert(tot_ctx_acc[i], 100 * correct_ctxs[i] / #test)
  end
  table.insert(tot_acc_rbase, 100 * corrects_rbase / #test)
  table.insert(tot_acc_base, 100 * corrects_base / #test)
  table.insert(tot_acc_top, 100 * corrects_top / #test)
end

if #flags.title > 0 then
  local empty = string.rep('#', #flags.title)
  print(string.format('###%s###', empty))
  print(string.format('#  %s  #', flags.title))
  print(string.format('###%s###', empty))
  print()
end

print 'Configuration'
print '----------------'
print(string.format('               Classifier  |  NN'))
print(string.format('                  Dataset  |  %s', dataset))
print(string.format('                   Target  |  %s', target))
print(string.format('          Context feature  |  %s', ctx_feature))
print(string.format('               # contexts  |  %d', #ctx_feature_values))
print(string.format('              Window size  |  %d', window_size))
print(string.format('         Test stream size  |  %d', stream_size))

print(string.format('           Concept length  |  %d', context_length))
print(string.format('       Concept recurrence  |  %d', number_of_contexts))
print(string.format('             # iterations  |  %d', flags.it))

for i, v in ipairs(types_detections) do
  print '----------------'
  print(v)
  print(string.format('        Context accuracy: %.2f%% (sdev: %.2f)', Mean(tot_ctx_acc[i])))
  print(string.format(' Classification accuracy: %.2f%% (sdev: %.2f)', Mean(tot_acc[i]))) 
end
print '----------------'
print(string.format(' Single Clas. Base. acc.: %.2f%% (sdev: %.2f)', Mean(tot_acc_base)))
print(string.format('  Rand. Clas. Base. acc.: %.2f%% (sdev: %.2f)', Mean(tot_acc_rbase)))
print(string.format('        Topline accuracy: %.2f%% (sdev: %.2f)', Mean(tot_acc_top)))
print '----------------'
print()