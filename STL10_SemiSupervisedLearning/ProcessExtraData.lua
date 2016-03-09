require 'nn'
require 'image'
require 'xlua'
print('1')
torch.setdefaulttensortype('torch.FloatTensor')
print('2')
local extradata = torch.class 'extradata'
print('3')
function extradata:__init(full)
	filename = 'stl-10/extra.t7b'
	numSamples = 70000
	numChannels = 3
	height = 96
	width = 96
        print('4')  
	extrasize = numSamples
        print('5')
	self.extraData = {
	     data = torch.Tensor(extrasize, numChannels, height, width),
	     size = function() return extrasize end
	}
	print('6')
	raw_table = torch.load(filename)
	-- load raw data
        print('7')
        print(self.extraData.data[1]:size())
        print('-------------')
        print(raw_table.data[1][1]:size()) 
	for i=1 , extrasize do
                print(i) 
		self.extraData.data[i]:copy(raw_table.data[1][i]:float())
	end
        print('8')
	self.extraData.data = self.extraData.data:float()
                         
	collectgarbage()
        print('9')
end


function extradata:normalize()

  local extraData = self.extraData
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,extraData:size() do
     xlua.progress(i, extraData:size())
     -- rgb -> yuv
     local rgb = extraData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     extraData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = extraData.data:select(2,2):mean()
  local std_u = extraData.data:select(2,2):std()
  extraData.data:select(2,2):add(-mean_u)
  extraData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = extraData.data:select(2,3):mean()
  local std_v = extraData.data:select(2,3):std()
  extraData.data:select(2,3):add(-mean_v)
  extraData.data:select(2,3):div(std_v)

  extraData.mean_u = mean_u
  extraData.std_u = std_u
  extraData.mean_v = mean_v
  extraData.std_v = std_v


end

function extradata:whiten()
	self.extraData.data = unsup.zca_whiten(self.extraData.data)[1]
end


