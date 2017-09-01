require 'numo/narray'
require 'pp'
require 'wrong/assert'
include Wrong::Assert

def sigmoid(x)
    1.0 / (1 + Numo::NMath.exp(-x))
end

def sigmoid_derivative(values)
    values * (1 - values)
end

def tanh_derivative(values)
    1.0 - values**2
end

# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args)
	Numo::DFloat.new(*args).rand * (b - a) + a
end

class LstmParam
    attr_accessor :mem_cell_ct, :x_dim, :wg, :wi, :wf, :wo, :bg, :bi, :bf, :bo, :wg_diff, :wi_diff, :wf_diff, :wo_diff, 
					:bg_diff, :bi_diff, :bf_diff, :bo_diff
    def initialize(mem_cell_ct, x_dim)
	@mem_cell_ct = mem_cell_ct
        @x_dim = x_dim
        puts @concat_len = x_dim + mem_cell_ct

        # weight matrices
        @wg = rand_arr(-0.1, 0.1, @mem_cell_ct, @concat_len)
        @wi = rand_arr(-0.1, 0.1, @mem_cell_ct, @concat_len) 
        @wf = rand_arr(-0.1, 0.1, @mem_cell_ct, @concat_len)
        @wo = rand_arr(-0.1, 0.1, @mem_cell_ct, @concat_len)

        # bias terms
        @bg = rand_arr(-0.1, 0.1, @mem_cell_ct) 
        @bi = rand_arr(-0.1, 0.1, @mem_cell_ct) 
        @bf = rand_arr(-0.1, 0.1, @mem_cell_ct) 
        @bo = rand_arr(-0.1, 0.1, @mem_cell_ct) 

        # diffs (derivative of loss function w.r.t. all parameters)
        @wg_diff = Numo::DFloat.zeros(@mem_cell_ct, @concat_len) 
        @wi_diff = Numo::DFloat.zeros(@mem_cell_ct, @concat_len) 
        @wf_diff = Numo::DFloat.zeros(@mem_cell_ct, @concat_len) 
        @wo_diff = Numo::DFloat.zeros(@mem_cell_ct, @concat_len)
        @bg_diff = Numo::DFloat.zeros(@mem_cell_ct) 
        @bi_diff = Numo::DFloat.zeros(@mem_cell_ct) 
        @bf_diff = Numo::DFloat.zeros(@mem_cell_ct) 
	@bo_diff = Numo::DFloat.zeros(@mem_cell_ct) 
    end

    def apply_diff(lr = 1)
        @wg -= lr * @wg_diff
        @wi -= lr * @wi_diff
        @wf -= lr * @wf_diff
        @wo -= lr * @wo_diff
        @bg -= lr * @bg_diff
        @bi -= lr * @bi_diff
        @bf -= lr * @bf_diff
        @bo -= lr * @bo_diff

        # reset diffs to zero of similar shape
        @wg_diff = @wg.new_zeros
        @wi_diff = @wi.new_zeros 
        @wf_diff = @wf.new_zeros 
        @wo_diff = @wo.new_zeros 
        @bg_diff = @bg.new_zeros
        @bi_diff = @bi.new_zeros 
        @bf_diff = @bf.new_zeros 
	@bo_diff = @bo.new_zeros 
    end
end

class LstmState
    attr_accessor :g, :i, :f, :o, :s, :h, :bottom_diff_h, :bottom_diff_s
    def initialize(mem_cell_ct, _x_dim)
        @g = Numo::DFloat.zeros(mem_cell_ct)
        @i = Numo::DFloat.zeros(mem_cell_ct)
        @f = Numo::DFloat.zeros(mem_cell_ct)
        @o = Numo::DFloat.zeros(mem_cell_ct)
        @s = Numo::DFloat.zeros(mem_cell_ct)
        @h = Numo::DFloat.zeros(mem_cell_ct)
        @bottom_diff_h = @h.new_zeros
	@bottom_diff_s = @s.new_zeros
    end
end

class LstmNode
    attr_accessor :state, :param, :xc, :s_prev, :h_prev
    def initialize(lstm_param, lstm_state)
        # store reference to parameters and to activations
        @state = lstm_state
        @param = lstm_param
        # non-recurrent input concatenated with recurrent input
	@xc = nil
    end

    def bottom_data_is(x, s_prev = nil, h_prev = nil)
        # if this is the first lstm node in the network
        if s_prev.nil? then s_prev = @state.s.new_zeros end
        if h_prev.nil? then h_prev = @state.h.new_zeros end
        
        # save data for use in backprop
        @s_prev = s_prev
        @h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = Numo::NArray.hstack([x, h_prev])
        @state.g = Numo::NMath.tanh((@param.wg.dot xc) + @param.bg)
        @state.i = sigmoid((@param.wi.dot xc) + @param.bi)
        @state.f = sigmoid((@param.wf.dot xc) + @param.bf)
        @state.o = sigmoid((@param.wo.dot xc) + @param.bo)
        @state.s = @state.g * @state.i + s_prev * @state.f
        @state.h = @state.s * @state.o

	@xc = xc
    end

    def top_diff_is(top_diff_h, top_diff_s)
        # notice that top_diff_s is carried along the constant error carousel
        ds = @state.o * top_diff_h + top_diff_s
        dopt = @state.s * top_diff_h
        di = @state.g * ds
        dg = @state.i * ds
        df = @s_prev * ds

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(@state.i) * di 
        df_input = sigmoid_derivative(@state.f) * df 
        do_input = sigmoid_derivative(@state.o) * dopt
        dg_input = tanh_derivative(@state.g) * dg

        # diffs w.r.t. inputs
        @param.wi_diff += di_input.outer(@xc)
        @param.wf_diff += df_input.outer(@xc)
        @param.wo_diff += do_input.outer(@xc)
        @param.wg_diff += dg_input.outer(@xc)
        @param.bi_diff += di_input
        @param.bf_diff += df_input       
        @param.bo_diff += do_input
        @param.bg_diff += dg_input       

        # compute bottom diff
        dxc = @xc.new_zeros
        dxc += (@param.wi.transpose).dot di_input
        dxc += (@param.wf.transpose).dot df_input
        dxc += (@param.wo.transpose).dot do_input
        dxc += (@param.wg.transpose).dot dg_input

        # save bottom diffs
        @state.bottom_diff_s = ds * @state.f
        @state.bottom_diff_h = dxc[@param.x_dim..-1]
    end
end

class LstmNetwork
    attr_accessor :lstm_param, :lstm_node_list, :x_list
    def initialize(lstm_param)
        @lstm_param = lstm_param
        @lstm_node_list = []
        # input sequence
	@x_list = []
    end

    def y_list_is(y_list, loss_layer)
        # """
        # Updates diffs by setting target sequence 
        # with corresponding loss layer. 
        # Will *NOT* update parameters.  To update parameters,
        # call @lstm_param.apply_diff()
        # """

        @lstm_param.apply_diff

	# Provided by gem 'wrong'. You can roll your own, but this paints a nice report.
	assert { (y_list.size) == (@x_list.size) }

	idx = (@x_list.size) - 1
	# first node only gets diffs from label ...
	loss = loss_layer.loss(@lstm_node_list[idx].state.h, y_list[idx])
	diff_h = loss_layer.bottom_diff(@lstm_node_list[idx].state.h, y_list[idx])
	# here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = Numo::DFloat.zeros(@lstm_param.mem_cell_ct)
        @lstm_node_list[idx].top_diff_is(diff_h, diff_s)
	idx -= 1

	while idx >= 0
	    loss += loss_layer.loss(@lstm_node_list[idx].state.h, y_list[idx])
            diff_h = loss_layer.bottom_diff(@lstm_node_list[idx].state.h, y_list[idx])
            diff_h += @lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = @lstm_node_list[idx + 1].state.bottom_diff_s
            @lstm_node_list[idx].top_diff_is(diff_h, diff_s)
	    idx -= 1 
	end

	    loss
    end

    def x_list_clear
	@x_list = []
    end

    def x_list_add(x)
        @x_list.push(x)
        if (@x_list.size) > (@lstm_node_list.size)
            # need to add new lstm node, create new state mem
            lstm_state = LstmState.new(@lstm_param.mem_cell_ct, @lstm_param.x_dim)      
	    @lstm_node_list.push(LstmNode.new(@lstm_param, lstm_state))
	end

	# get index of most recent x input
	idx = (@x_list.size) - 1

	if idx.zero?
            # no recurrent inputs yet
            @lstm_node_list[idx].bottom_data_is(x)
        else
            s_prev = @lstm_node_list[idx - 1].state.s
            h_prev = @lstm_node_list[idx - 1].state.h
            @lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)
        end

    end

end









# TESTING LSTM














class ToyLossLayer
    # """
    # Computes square loss with first element of hidden layer array.
    # """

    def self.loss(pred, label)
        (pred[0] - label)**2
    end

    def self.bottom_diff(pred, label)
        diff = pred.new_zeros
        diff[0] = 2 * (pred[0] - label)
	diff
    end
end




def example_0
    # parameters for input data dimension and lstm cell count 
    mem_cell_ct = 100
    x_dim = 50
    lstm_param = LstmParam.new(mem_cell_ct, x_dim) 
    lstm_net = LstmNetwork.new(lstm_param)
    y_list = [-0.5, 0.2, 0.1, -0.5]
    input_val_arr = Array.new(y_list.size) { Numo::DFloat.new(x_dim).rand }

	100.times do |cur_iter|
		pp "cur iter: " + cur_iter.inspect
		y_list.size.times do |ind|
			lstm_net.x_list_add(input_val_arr[ind])
			pp "y_pred[" + ind.inspect + "] : " + (lstm_net.lstm_node_list[ind].state.h[0]).inspect
		end

		loss = lstm_net.y_list_is(y_list, ToyLossLayer)
		pp "loss: " + loss.inspect
        lstm_param.apply_diff(lr = 0.1)
		lstm_net.x_list_clear

	end
end


if __FILE__ == $0
    example_0    
end
