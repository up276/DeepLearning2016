-- define data augmentation modules
-- inspired from : https://github.com/nagadomi/kaggle-cifar10-torch7/blob/cuda-convnet2/lib/data_augmentation.lua

require 'image'
require 'xlua'

-- horizontal flip module
-- this is the initial data augmentation code provided in the assignement

do
  local BatchHFlip,parent = torch.class('nn.BatchHFlip', 'nn.Module')

  function BatchHFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchHFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      new_input = input:clone()
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then 
          new_input[i] = image.hflip(input[i])
        else new_input[i] = input[i] 
        end
      end
    end
    self.output:set(new_input)
    return self.output
  end
end

-- horizontal flip module

do
  local BatchVFlip,parent = torch.class('nn.BatchVFlip', 'nn.Module')

  function BatchVFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchVFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      new_input = input:clone()
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then 
          new_input[i] = image.vflip(input[i]) 
        else new_input[i] = input[i]
        end
      end
    end
    self.output:set(new_input)
    return self.output
  end
end

-- Translate module

do
  local BatchTranslate,parent = torch.class('nn.BatchTranslate', 'nn.Module')

  function BatchTranslate:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchTranslate:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local patchsize = input:size(3)
      new_input = input:clone()
      local flip_mask = torch.randperm(bs):le(bs/2)
      local dist = math.floor((torch.rand(1)[1] * 0.4 - 0.2) * patchsize)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then
          new_input[i] = image.translate(input[i],dist,dist)
        else new_input[i] = input[i]
        end
      end
    end
    self.output:set(new_input)
    return self.output
  end
end

-- Rotate module

do
  local BatchRotate,parent = torch.class('nn.BatchRotate', 'nn.Module')

  function BatchRotate:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchRotate:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local patchsize = input:size(3) 
      local flip_mask = torch.randperm(bs):le(bs/2)
      -- generate random angle between -pi/6 and + pi/6
      local rad = torch.rand(1)[1] * math.pi * 2/6  - math.pi/6
      new_input = input:clone()
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then 
          new_input[i] = image.rotate(input[i],rad)
        else new_input[i] = input[i]
        end
      end
    end
    collectgarbage()
    self.output:set(new_input)
    return self.output
  end
end

-- Contrast module

do
  local BatchContrast,parent = torch.class('nn.BatchContrast', 'nn.Module')

  function BatchContrast:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchContrast:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local patchsize = input:size(3)
      new_input = input:clone()
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then 
            local factors = torch.rand(1, 3):mul(1.5):add(0.5)
            local unfolded = input[i]:reshape(3, patchsize*patchsize):transpose(1, 2)
            ce, cv = unsup.pcacov(unfolded)
            local proj = unfolded * cv
            proj:cmul(torch.expand(factors, patchsize*patchsize, 3))
            new_input[i] = (proj * torch.inverse(cv)):transpose(1, 2):reshape(3, patchsize, patchsize)
         else new_input[i] = input[i]
         end
      end
    end
    self.output:set(new_input)
    return self.output
  end
end

-- Crop module

do
  local BatchCrop,parent = torch.class('nn.BatchCrop', 'nn.Module')

  function BatchCrop:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchCrop:updateOutput(input)
    if self.train then 
      local bs = input:size(1)
      local feature_maps = input:size(2)
      local patchsize = input:size(3)
      -- local flip_mask = torch.randperm(bs):le(bs/2) -- randomness already obtained via torch.rand(1)[1] > 1/3
      new_input = input:clone()
      for i=1,input:size(1) do
        -- if flip_mask[i] == 1 then 
          local factor = torch.rand(1)[1]
          local threshold = 1/2
          -- Crop only for significantly big crops
          if factor > threshold then
            local crop_size = math.floor(patchsize * factor)
            -- determine region to crop on the image
            local start_cropping_rightleft_factor = torch.rand(1)[1]
            local start_cropping_updown_factor = torch.rand(1)[1]
            local start_cropping_rightleft = math.floor(start_cropping_rightleft_factor*(patchsize - crop_size))
            local start_cropping_updown = math.floor(start_cropping_updown_factor*(patchsize - crop_size))
            temp = torch.Tensor(feature_maps, crop_size, crop_size):fill(0)
            temp = image.crop(input[i], start_cropping_rightleft, start_cropping_updown, start_cropping_rightleft + crop_size, start_cropping_updown + crop_size)
            new_input[i] = image.scale(temp, patchsize, patchsize)
          else new_input[i] = input[i] 
          end
        -- end
      end
      self.output:set(new_input)
      return self.output
    end
  end
end

-- TODO - ADAPT CODE
--[[
require './save_images'
local x = torch.load("../stl-10/train_x.bin") -- Change to our train x data
local y = torch.load("../stl-10/train_y.bin") -- Change to our train y data
local target = 13
local jit = data_augmentation(x[target]:resize(1, 3, 32, 32), y[target]:resize(1, 10))
print(jit:size(1))
save_images(jit, jit:size(1), "jitter.png")
--]]
