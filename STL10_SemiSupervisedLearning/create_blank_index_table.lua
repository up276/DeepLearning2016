--s script implements load the previous saved model and
-- generate prediction on test set
----------------------------------------------------------------------
require 'torch'
require 'nn'
require 'image'
require 'optim'
--require 'csvigo'
dofile 'ProcessExtraData.lua'


--trsize = 5000
channel = 3
height = 96
width = 96
nExtradata = 10000

-- load previous saved model
print '==> loading trained model'
--model = torch.load('model.net')

print '==> loading extradata data'
--load extradata
extraData = torch.load('extraDataBatch.t7').extraData
extraData.data = extraData.data:float()


result ={
     index = torch.ByteTensor(extraData.data:size(1)):fill(0),  -- this is a mask
     label = {}
}

local ix = 1  --count how many data have we used

function predict()
  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> predict on extradata set:')

  local bs = 25
  for i=1,extraData.data:size(1),bs do
    -- i indicate current index

        local pred = model:forward(extraData.data:narrow(1,i,bs))

        -- get the peudo_label: t, label, score
        for e =1 ,bs do  -- each instance
          local count =0
          score = pred[e]:max()
          for it=1, 10 do    
             if pred[e][it] == score then
                itlabel =it
             end 
          end

          -- fill result table i
          if score > 16 then    
              print('instance '..(i+e-1)..'score'..score)  -- which instance
              --print(ix)
              -- if belongs to two class, drop it
               result.index[i+e-1] =  1
              -- print(result.index[(i-1)*bs+e])
               table.insert(result.label,itlabel)
               ix = ix +1
          end
        end
  end
end

--predict()
print('select '..#result.label)
torch.save('BatchlabelIx.t7',result)

