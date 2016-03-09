require 'torch'
require 'nn'
require 'image'
require 'xlua'
require 'io'
require 'optim'
require 'cunn'
require './save_images.lua'
dofile 'provider70K.lua'
local c = require 'trepl.colorize'


----------------------------------------------------------------------

collectgarbage()

print '==> processing options'

cmd = torch.CmdLine()
opt = cmd:parse(arg or {})

----------------------------------------------------------------------
print '==> defining some tools'
local trsize = 4000
local tesize = 8000
local channel = 3
local height = 96
local width = 96
-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}
print '1'
-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
print '2'
-- Load model
model = torch.load('logs/sample_model_with_data_aug_psdo_lable2/model.net')
provider = torch.load('provider70K_aug.t7')
--provider = torch.load('providersemisupbatch.t7')
print '3'
-- Log results to files
testLogger = optim.Logger(paths.concat('logs/sample_model_with_data_aug_psdo_lable2/test70K_aug.log'))
print '4'
-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end
print '5'
----------------------------------------------------------------------

torch.setdefaulttensortype('torch.FloatTensor')

-- parse STL-10 data from table into Tensor
function parseDataLabel(d, numSamples, numChannels, height, width)
   t = torch.ByteTensor(numSamples, numChannels, height, width)
   l = torch.ByteTensor(numSamples)
   idx = 1
   for i = 1, #d do
      this_d = d[i]
      for j = 1, #this_d do
    	t[idx]:copy(this_d[j])
    	l[idx] = i
    	idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t, l
end



----------------------------------------------------------------------

function load_test_data()

	--if not paths.dirp('stl-10') then
	--     os.execute('mkdir stl-10')
	 --    local www = {
		 --train = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/train.t7b',
		 --val = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/val.t7b',
		 --extra = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/extra.t7b',
		 --test = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/test.t7b'
	  --   }

	     --os.execute('wget ' .. www.train .. '; '.. 'mv train.t7b stl-10/train.t7b')
	     --os.execute('wget ' .. www.val .. '; '.. 'mv val.t7b stl-10/val.t7b')
	     --os.execute('wget ' .. www.test .. '; '.. 'mv test.t7b stl-10/test.t7b')
	     --os.execute('wget ' .. www.extra .. '; '.. 'mv extra.t7b stl-10/extra.t7b')
	--end
	--train = torch.load('stl-10/train.t7b')
	--raw_train = torch.load('stl-10/parsed_train.t7b')
	raw_test = torch.load('stl-10/test.t7b')

	-- load and parse dataset
	--trainData = {
	--     data = torch.Tensor(),
	--     labels = torch.Tensor(),
	--     size = function() return trsize end
        --}
	--trainData.data, trainData.labels = parseDataLabel(train.data,
	--		                                   trsize, channel, height, width)
	--trainData.data = trainData.data:float()
        --trainData.labels = trainData.labels:float()
	--local trainData = trainData
	
	-- load and parse dataset
	testData = {
	     data = torch.Tensor(),
	     labels = torch.Tensor(),
	     size = function() return tesize end
	}
	testData.data, testData.labels = parseDataLabel(raw_test.data,
	 	                                           tesize, channel, height, width)
	testData.data = testData.data:float()
	testData.labels = testData.labels:float()
	--local testData = testData
	--testData.data = testData.data:float()
        --testData.labels = testData.labels:float()
	--trainData.data = trainData.data:float()
	--trainData.labels = trainData.labels:float()
	--testData.data = testData.data:float()
	--testData.labels = testData.labels:float()
	--collectgarbage()

	----------------------------------------------------------------------
	  -- preprocess/normalize train/val sets
	  --
	--local trainData = self.trainData
	--local testData = self.testData

	print '<trainer> preprocessing data (color space + normalization)'
	collectgarbage()

	-- preprocess trainSet
	local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
	--for i = 1,trainData:size() do
	--     xlua.progress(i, trainData:size())
	     -- rgb -> yuv
	--     local rgb = trainData.data[i]
	--     local yuv = image.rgb2yuv(rgb)
	     -- normalize y locally:
	--     yuv[1] = normalization(yuv[{{1}}])
	--     trainData.data[i] = yuv
	--end
	-- normalize u globally:
	--local mean_u = trainData.data:select(2,2):mean()
	--local std_u = trainData.data:select(2,2):std()
	--trainData.data:select(2,2):add(-mean_u)
	--trainData.data:select(2,2):div(std_u)
	-- normalize v globally:
	--local mean_v = trainData.data:select(2,3):mean()
	--local std_v = trainData.data:select(2,3):std()
	--trainData.data:select(2,3):add(-mean_v)
	--trainData.data:select(2,3):div(std_v)

	--trainData.mean_u = mean_u
	--trainData.std_u = std_u
	--trainData.mean_v = mean_v
	--trainData.std_v = std_v
	--provider = torch.load('providersemisupbatch.t7')

	local mean_u = provider.trainData.data:select(2,2):mean()
        local std_u = provider.trainData.data:select(2,2):std()
        local mean_v = provider.trainData.data:select(2,3):mean()
        local std_v = provider.trainData.data:select(2,3):std()
	  -- get train mean
  	--local mean_u = provider.mean_u
  	print("mean_u",mean_u)
	--local std_u = provider.std_u
 	print("std_u",std_u)
  	--local mean_v = provider.mean_u
 	print("mean_v",mean_v)
  	--local std_v = provider.std_v
	print("std_v",std_v)

	-- preprocess testSet
	for i = 1,testData:size() do
	    xlua.progress(i, testData:size())
	     -- rgb -> yuv
	    local rgb = testData.data[i]
	    local yuv = image.rgb2yuv(rgb)
	    -- normalize y locally:
	    yuv[{1}] = normalization(yuv[{{1}}])
	    testData.data[i] = yuv
	end
	-- normalize u globally:
	testData.data:select(2,2):add(-mean_u)
	testData.data:select(2,2):div(std_u)
	-- normalize v globally:
	testData.data:select(2,3):add(-mean_v)
	testData.data:select(2,3):div(std_v)
	
	----------------------------------------------------------------------
	print '==> verify statistics'

	-- It's always good practice to verify that data is properly
	-- normalized.
	print('train size - ',provider.trainData.data:size())
	trainMean = provider.trainData.data[{ {},1}]:mean() 
	trainStd = provider.trainData.data[{ {},1}]:std()
	print("----------------------------------")
	testMean = testData.data[{ {},1 }]:mean()
	testStd = testData.data[{ {},1 }]:std()

	print('test data mean: ' .. testMean)
	print('test data standard deviation: ' .. testStd)

end	
	
function test()
	print 'test'
	-- local vars
	local time = sys.clock()

	-- averaged param use?
	--if average then
	 -- cachedparams = parameters:clone()
	 -- parameters:copy(average)
	--end

	-- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
	model:evaluate()

	-- test over test data
	print('==> testing on test set:')
        f = io.open('predictions70K_aug.csv', 'w')
	f:write('Id,Prediction' .. '\n')
	local bs = 25

	-- code to get first layer output

	--firstLayer = model:get(1)
	--print("firstLayer:",firstLayer)
        --print("testdata size:",testData.data:size())
	--opt = firstLayer:forward(testData.data:narrow(1,1,1):cuda())
	--opt1 = opt:narrow(1,1,1)
	--print("output of the first layer")
	--print(opt1)
	--save_images(opt, opt:size(2), "firstLayerOutput.png")
	
	-- code complete - to get firs layer output

	-- comments start
 
	for i=1,testData.data:size(1),bs do
	    local outputs = model:forward(testData.data:narrow(1,i,bs):cuda())
	    for j=1,25 do
		pred = outputs[j]
		--print('output',j,'th row : ', pred)
		b,index = torch.max(pred,1)
                --print('index', index[1])
                --print('b', b)
                --print(index)
		t = i+j-1
                f:write(t..','..index[1]..'\n')
	
	    end
          
	    confusion:batchAdd(outputs, testData.labels:narrow(1,i,bs))
	end
        
	confusion:updateValids()
	print('val accuracy:', confusion.totalValid * 100)


	-- comments ends

	--for t = 1,testData:size() do
	  -- disp progress
	  --xlua.progress(t, testData:size())

	  -- get new sample
	  --local input = testData.data[t]
	  --input = input:double()
	  --local target = testData.labels[t]

	  -- test sample
	  --local pred = model:forward(input)
	  --confusion:add(pred, target)
	  --f = io.open('predictions.csv', 'w')
	  --f:write(a[i] .. '\n')  -- You should know that \n brings newline and .. concats stuff
	  -- b,index = torch.max(pred,1)
	  --print(index[1])
	  --print(b)
	  --print(index)
	  --f:write(t..','..index[1] .. '\n')	
	  --confusion:add(pred, target)
	  --print(pred)
	--end
	f:close()
	--print 'predictions are as below'	
	--print(pred)
	-- timing
	time = sys.clock() - time
	time = time / testData:size()
	print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	print(confusion)
	testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
	confusion:zero()

	-- update log/plot
	--testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
	--if opt.plot then
	--testLogger:style{['% mean class accuracy (test set)'] = '-'}
	--testLogger:plot()
	--end

	-- next iteration:
	--confusion:zero()
end
	
function create_prediction()
	print 'hi'
	--dofile 'doall.lua'
	--local filename = paths.concat('results', 'model.net')
	--print filename
	--model = torch.load('results/model.net')
	load_test_data()
	--dofile '5_test.lua'
	print(testData:size())
	test()

end

create_prediction()
