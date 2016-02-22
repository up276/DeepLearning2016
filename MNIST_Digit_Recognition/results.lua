require 'torch'
require 'nn'
require 'image'
require 'xlua'
require 'io'
require 'optim'
----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
opt = cmd:parse(arg or {})

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Load model
model = torch.load('results/model.net')

-- Log results to files
testLogger = optim.Logger(paths.concat('results/test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------

function load_test_data()
	tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

	data_path = 'mnist.t7'
	train_file = paths.concat(data_path, 'train_32x32.t7')
	test_file = paths.concat(data_path, 'test_32x32.t7')
         
	if not paths.filep(train_file) or not paths.filep(test_file) then
	   os.execute('wget ' .. tar)
	   os.execute('tar xvf ' .. paths.basename(tar))
	end


	----------------------------------------------------------------------
	print '==> loading dataset'
        tesize = 10000
	
	loaded = torch.load(train_file, 'ascii')
	trainData = {
	   data = loaded.data,
	   labels = loaded.labels,
	   size = function() return trsize end
	}

	loaded = torch.load(test_file, 'ascii')
	testData = {
	   data = loaded.data,
	   labels = loaded.labels,
	   size = function() return tesize end
	}
	----------------------------------------------------------------------
	print '==> preprocessing data'

	-- Preprocessing requires a floating point representation (the original
	-- data is stored on bytes). Types can be easily converted in Torch,
	-- in general by doing: dst = src:type('torch.TypeTensor'),
	-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
	-- for simplicity (float(),double(),cuda(),...):

	trainData.data = trainData.data:float()
	testData.data = testData.data:float()

	-- We now preprocess the data. Preprocessing is crucial
	-- when applying pretty much any kind of machine learning algorithm.

	-- For natural images, we use several intuitive tricks:
	--   + images are mapped into YUV space, to separate luminance information
	--     from color information
	--   + the luminance channel (Y) is locally normalized, using a contrastive
	--     normalization operator: for each neighborhood, defined by a Gaussian
	--     kernel, the mean is suppressed, and the standard deviation is normalized
	--     to one.
	--   + color channels are normalized globally, across the entire dataset;
	--     as a result, each color component has 0-mean and 1-norm across the dataset.

	-- Convert all images to YUV

	-- As we are using MNIST which only has one channel, ignore the above paragraph

	-- Normalize each channel, and store mean/std.
	-- These values are important, as they are part of
	-- the trainable parameters. At test time, test data will be normalized
	-- using these values.
	print '==> preprocessing data: normalize globally'
	mean = trainData.data[{ {},1,{},{} }]:mean()
	std = trainData.data[{ {},1,{},{} }]:std()
	
	-- Normalize test data, using the training means/stds
	testData.data[{ {},1,{},{} }]:add(-mean)
	testData.data[{ {},1,{},{} }]:div(std)

	-- Local normalization
	print '==> preprocessing data: normalize locally'
	--
	-- -- Define the normalization neighborhood:
	neighborhood = image.gaussian1D(7)
	--
	-- -- Define our local normalization operator (It is an actual nn module,
	-- -- which could be inserted into a trainable model):
	normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
	--
	-- -- Normalize all channels locally:
	for i = 1,testData:size() do
	    testData.data[{ i,{1},{},{} }] = normalization:forward(testData.data[{ i,{1},{},{} }])
	end
	--    for i = 1,valData:size() do
	--       valData.data[{ i,{1},{},{} }] = normalization:forward(valData.data[{ i,{1},{},{} }])
	--       end
	--       --for i = 1,testData:size() do
	--       --   testData.data[{ i,{1},{},{} }] = normalization:forward(testData.data[{ i,{1},{},{} }])
	--       --end
	--
	--
	
	----------------------------------------------------------------------
	print '==> verify statistics'

	-- It's always good practice to verify that data is properly
	-- normalized.


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
        f = io.open('predictions.csv', 'w')
	f:write('Id,Predition' .. '\n')
	for t = 1,testData:size() do
	  -- disp progress
	  xlua.progress(t, testData:size())

	  -- get new sample
	  local input = testData.data[t]
	  input = input:double()
	  local target = testData.labels[t]

	  -- test sample
	  local pred = model:forward(input)
	  confusion:add(pred, target)
	  --f = io.open('predictions.csv', 'w')
	  --f:write(a[i] .. '\n')  -- You should know that \n brings newline and .. concats stuff
	  b,index = torch.max(pred,1)
	  --print(index[1])
	  --print(b)
	  --print(index)
	  f:write(t..','..index[1] .. '\n')	
	  --confusion:add(pred, target)
	  --print(pred)
	end
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
