-- ref : https://github.com/nagadomi/kaggle-cifar10-torch7/blob/cuda-convnet2/lib/save_images.lua

require 'torch'
require 'image'

function save_images(x, n, file_name)
   file_name = file_name or "./out.png"
   local input = x:narrow(2, 1, n)
   local view = image.toDisplayTensor({input = input,
				       padding = 2,
				       nrow = 9,
				       symmetric = true})
   image.save(file_name, view)
end

return true
