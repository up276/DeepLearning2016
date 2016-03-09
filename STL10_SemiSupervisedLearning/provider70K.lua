require 'nn'
require 'image'
require 'xlua'
dofile 'ProcessExtraData.lua'

torch.setdefaulttensortype('torch.FloatTensor')

-- parse STL-10 data from table into Tensor
function parseDataLabel(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local l = torch.ByteTensor(numSamples)
   local idx = 1
   for i = 1, #d do
      local this_d = d[i]
      for j = 1, #this_d do
    t[idx]:copy(this_d[j])
    l[idx] = i
    idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t, l
end


local Provider70K = torch.class 'Provider70K'

function Provider70K:__init(full)
  trsize = 4000
  local valsize = 1000  -- Use the validation here as the valing set
  --local extrasize = 100000
  local channel = 3
  local height = 96
  local width = 96

  -- download dataset
  if not paths.dirp('stl-10') then
     os.execute('mkdir stl-10')
     local www = {
         train = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/train.t7b',
         val = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/val.t7b',
         extra = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/extra.t7b',
         test = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/test.t7b'
     }

     os.execute('wget ' .. www.train .. '; '.. 'mv train.t7b stl-10/train.t7b')
     os.execute('wget ' .. www.val .. '; '.. 'mv val.t7b stl-10/val.t7b')
     os.execute('wget ' .. www.test .. '; '.. 'mv test.t7b stl-10/test.t7b')
     os.execute('wget ' .. www.extra .. '; '.. 'mv extra.t7b stl-10/extra.t7b')
  end

  local raw_train = torch.load('stl-10/train.t7b')
  local raw_val = torch.load('stl-10/val.t7b')
  --local raw_extra = torch.load('stl-10/extra.t7b')
  --local raw_train = torch.load('stl-10/parse_train.t7b')
  --local raw_val = torch.load('stl-10/parse_val.t7b')

  -- load and parse dataset
  self.trainData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return trsize end
  }
  self.trainData.data, self.trainData.labels = parseDataLabel(raw_train.data,
                                                   trsize, channel, height, width)
  local trainData = self.trainData
  self.valData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return valsize end
  }
  self.valData.data, self.valData.labels = parseDataLabel(raw_val.data,
                                                 valsize, channel, height, width)
  local valData = self.valData


  --self.extraData = {
  --   data = torch.Tensor(extrasize, numChannels, height, width),
  --   size = function() return extrasize end
  --	}
	
  --for i=1,extrasize do
  --   self.extraData.data[i] :copy(raw_table.data[1][i]:float())
  --end

  --self.extraData.data = self.extraData.data:float()

  --collectgarbage() 

  -- convert from ByteTensor to Float
  self.trainData.data = self.trainData.data:float()
  self.trainData.labels = self.trainData.labels:float()
  self.valData.data = self.valData.data:float()
  self.valData.labels = self.valData.labels:float()
  collectgarbage()
end

function Provider70K:useExtra()

  local extra = torch.load('label70K_aug_Ix.t7')  --index, label{} 
  local nExtra = #extra.label
  print("nextra",nExtra)
  print('going to add ..'..nExtra..'data to traindata')
  local extradata = torch.load('extraData70K.t7').extraData.data
  -- select by index
  -- c
  local add = torch.Tensor(nExtra,3,96,96)
  local ix=1
  for i=1, extradata:size(1) do
    if extra.index[i]==1 then
      add[ix] = extradata[i]
      ix =ix +1
    end
  end
  local label =torch.Tensor(extra.label)
  print("size_before :",trsize)
  trsize = trsize + nExtra
  print("size_after :",trsize)
  self.trainData.data = torch.cat(self.trainData.data,add,1)
  self.trainData.labels = torch.cat(self.trainData.labels,label,1)

end



function Provider70K:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/val sets
  --
  local trainData = self.trainData
  local valData = self.valData

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v

  -- preprocess valSet
  for i = 1,valData:size() do
    xlua.progress(i, valData:size())
     -- rgb -> yuv
     local rgb = valData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     valData.data[i] = yuv
  end
  -- normalize u globally:
  valData.data:select(2,2):add(-mean_u)
  valData.data:select(2,2):div(std_u)
  -- normalize v globally:
  valData.data:select(2,3):add(-mean_v)
  valData.data:select(2,3):div(std_v)
end