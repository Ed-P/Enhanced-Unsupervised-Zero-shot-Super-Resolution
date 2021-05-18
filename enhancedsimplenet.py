import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable

class EnhancedSimpleNet(nn.Module):
	def __init__(self,Y=True):
		super(simpleNet, self).__init__()
		d = 1
		if Y == False:
			d = 3
		channel=64
		channel2=128
		self.input = nn.Conv2d(in_channels=d, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=False)
        
		self.conv_3_1 = nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 3, stride = 1, padding = 1, bias = True)
		self.conv_3_2 = nn.Conv2d(in_channels = channel * 2, out_channels = channel * 2, kernel_size = 3, stride = 1, padding = 1, bias = True)
		self.conv_5_1 = nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 5, stride = 1, padding = 2, bias = True)
		self.conv_5_2 = nn.Conv2d(in_channels = channel * 2, out_channels = channel * 2, kernel_size = 5, stride = 1, padding = 2, bias = True)
		self.confusion = nn.Conv2d(in_channels = channel * 4, out_channels = channel, kernel_size = 1, stride = 1, padding = 0, bias = True)

		self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel2, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels=channel2, out_channels=channel2, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv3 = nn.Conv2d(in_channels=channel2, out_channels=channel2, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv4 = nn.Conv2d(in_channels=channel2, out_channels=channel2, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv5 = nn.Conv2d(in_channels=channel2, out_channels=channel2, kernel_size=3, stride=1, padding=1, bias=False)
		
		self.conv6 = nn.Conv2d(in_channels=channel2, out_channels=channel2, kernel_size=3, stride=1, padding=1, bias=False)

	
		self.output = nn.Conv2d(in_channels=channel2, out_channels=d, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n))

	def forward(self, x):
		residual = x
		inputs = self.input(self.relu(x))
		out = inputs
        #
		output_3_1 = self.relu(self.conv_3_1(out))
		output_5_1 = self.relu(self.conv_5_1(out))

		input_2 = torch.cat([output_3_1, output_5_1], 1)
		output_3_2 = self.relu(self.conv_3_2(input_2))
		output_5_2 = self.relu(self.conv_5_2(input_2))
		#output_3_3 = self.relu(self.conv_3_2(input_2))
		#output_5_3 = self.relu(self.conv_5_2(input_2))
		#output = torch.cat([output_3_3, output_5_3], 1)        
		output = torch.cat([output_3_2, output_5_2], 1)
		out = self.confusion(output)
        #       
        #
		#out = torch.add(out, inputs)
		#out = self.output(self.relu(out))    
		#out = torch.add(out, residual)
		#out=self.input(self.relu(out))
		#output = torch.add(output, identity_data)
        
		out = self.conv1(out)
		out = self.conv2(self.relu(out))
		out = self.conv3(self.relu(out))
		out = self.conv4(self.relu(out))
		out = self.conv5(self.relu(out))
		out = self.conv6(self.relu(out))
         #
		#out = torch.add(out, inputs)
        #
		out = self.output(self.relu(out))
        #
		out = torch.add(out, residual)
        
		return out
