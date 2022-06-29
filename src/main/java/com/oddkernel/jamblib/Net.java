package com.oddkernel.jamblib;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

public abstract class Net {
    private String name = null;
    protected static final String BIAS_KEY = "__BIAS_KEY__";

    private List<String> features = new ArrayList<>();
    private List<String> target = new ArrayList<>();
    private Map<String, Map<String, Double>> synapses = new HashMap<>();
    private boolean normalize = false;
    private List<Double> input_mean = new ArrayList<>();
    private List<Double> input_std = new ArrayList<>();
    private List<Double> output_mean = new ArrayList<>();
    private List<Double> output_std = new ArrayList<>();
    private boolean training = false;
    private boolean normalize_output = false;

    @Override
    public String toString() {
        String features = String.join(",", this.features);
        String target = String.join(",", this.target);
        String synapses = "GRAPH:\n" + String.join(",\n", prettyGraph(this.synapses));
        return features + "\n" + target + "\n" + synapses;
    }

    private List<String> prettyGraph(Map<String, Map<String, Double>> synapses) {
        return synapses.entrySet().stream().map(entry -> entry.getKey() + ":" + prettyGraphEntry(entry)).collect(Collectors.toList());
    }
    private String prettyGraphEntry(Entry<String, Map<String, Double>> synapses) {
        String content = synapses.getValue().entrySet().stream().map(entry -> entry.getKey() + ":" + String.valueOf(entry.getValue())).reduce((a,b) -> a + "," + b).orElse("");
        return String.format("{%s}",content);
    }

    public Net(String name) {
        this.name = name;
    }

    public ModelErrorStats computeStats(List<List<Double>> trainFeatures, List<List<Double>> trainTargets) {
        List<List<Double>> predictTargets = this.eval(trainFeatures, new HashMap<>());
        Double mse = NumJa.mse(predictTargets, trainTargets);
        return new ModelErrorStats(mse);
        
    }

    private List<List<Double>> eval(List<List<Double>> features, HashMap<String, List<Double>> historial) {
        return features.stream().map(feature -> this.eval(feature, historial)).collect(Collectors.toList());
    }

    private Map<String, Double> getFeaturesMap(List<String> features, List<Double> instance) {
        return NumJa.arange(features.size()).stream().collect(Collectors.toMap(i -> features.get(i), i -> instance.get(i)));
    }

    private Map<String, Double> getTargetsMap(List<String> targets, List<Double> instance) {
        return NumJa.arange(targets.size()).stream().collect(Collectors.toMap(i -> targets.get(i), i -> instance.get(i)));
    }


    public Double dfs(String cell, List<Double> instance, Map<String, List<Double>> historial) {
        // Check that input sizes match
        if (instance.size() != this.features.size()) {
            throw new IllegalStateException(String.format("Instance %d does not match input layer size (%d).", instance.size(), this.features.size()));
        }

        // Create dict with input values from instance
        Map<String, Double> featuresMap = getFeaturesMap(this.features, instance);

        if (featuresMap.containsKey(cell)) {
            Double feature = featuresMap.get(cell);
            int featureIndex;
            if (this.normalize) {
                featureIndex = this.features.indexOf(cell);
                feature = (feature - this.input_mean.get(featureIndex)) / this.input_std.get(featureIndex);
            }
            List<Double> entry = new ArrayList<>();
            entry.add(feature);
            entry.add(feature);
            historial.put(cell, entry);
            return feature;
        } else {
            Map<String, Double> synapse = this.synapses.get(cell);
            // Get bias
            double biasWeight = synapse.get(BIAS_KEY);
            // Weighted sum of ingoing synapses
            Double signal = synapse.entrySet().stream().filter(destinationCell -> !destinationCell.getKey().equals(BIAS_KEY)).map(destinationCell ->  destinationCell.getValue() * this.dfs(destinationCell.getKey(), instance, historial)).reduce((a,b) -> a+b).orElse(0.0);

            // Return output value
            double outputSignal = 0.0;
            if (this.target.contains(cell)) {
                outputSignal = this.finalTransferFunction(biasWeight + signal);
                if (this.normalize_output) {
                    int featureIndex = this.target.indexOf(cell);
                    outputSignal = outputSignal*this.output_std.get(featureIndex) + this.output_mean.get(featureIndex);
                }
            } else {
                outputSignal = this.transferFunction(biasWeight + signal);
            }
            // Add values to history
            List<Double> entry = new ArrayList<>();
            entry.add(biasWeight + signal);
            entry.add(outputSignal);
            historial.put(cell, entry);
            return outputSignal;
        }

    }

    public List<Double> eval(List<Double> instance, Map<String, List<Double>> historial) {
        return this.target.stream().map( y -> this.dfs(y, instance, historial)).collect(Collectors.toList());
    }

    public void randomizeSynapses(double low, double high) {
        double amplitude = high - low;
        // For each synapse in the net
        for (Entry<String, Map<String, Double>> destinationNode : this.synapses.entrySet()) {
            String destinationNodeName = destinationNode.getKey();
            Map<String, Double> sources = destinationNode.getValue();
            for (Entry<String, Double> source : sources.entrySet()) {
                String sourceNodeName = source.getKey();
                // Assume weight 0
                double synapseWeight = 0.0;
                while (synapseWeight == 0.0) {
                    synapseWeight = Math.random() * amplitude + low;
                }
                this.synapses.get(destinationNodeName).put(sourceNodeName, synapseWeight);

            }
        }
    }

    public void addCells(String layerName, Integer layerSize) {
        //String cellName = String.format("[%s] Add layer '%s': size %d", this.name, layerName, layerSize);
        for (int layerIndex = 0; layerIndex < layerSize; layerIndex++) {
            this.addCell(layerName, layerIndex + 1, null);
        }

    }

    public void addCell(String basename, Integer layerIndex, String type) {
        String cellName = String.format("%s_%d", basename, layerIndex);
        // Create full cell name
        // Check if cell already exists
        if (this.synapses.containsKey(cellName)) {
            throw new IllegalStateException(String.format(
                    "Cell '%s'' already exists in net '%s'", cellName, this.name));
        }

        // Initialize cell synapses dict
        this.synapses.put(cellName, new HashMap<>());
        // Initialize bias to 0
        this.synapses.get(cellName).put(BIAS_KEY, 0.0);
        // Mark as input or output cell, if any
        if ("input".equals(type)) {
            this.features.add(cellName);
            this.input_mean.add(0.0);
            this.input_std.add(1.0);
        }
        else if ("output".equals(type)) {
            this.target.add(cellName);
            this.output_mean.add(0.0);
            this.output_std.add(1.0);
        }

    }

    public void addSynapse(String source, String destination, double weight) {
        if (!this.synapses.containsKey(source)) {
            throw new IllegalStateException(String.format("Cell '%s' does not exist in the net.", source));
        }
        if (!this.synapses.containsKey(destination)) {
            throw new IllegalStateException(String.format("Cell '%s' does not exist in the net.", destination));
        }
        this.synapses.get(destination).put(source, weight);
    }

    public void addSynapses(String source, String destination, Double weight, int input_dimension,
            int output_dimension) {
        for (int indexInput = 0; indexInput < input_dimension; indexInput++) {
            String sourceName = String.format("%s_%d", source, indexInput + 1);
            for (int indexOutput = 0; indexOutput < output_dimension; indexOutput++) {
                String destinationName = String.format("%s_%d", destination, indexOutput + 1);
                this.addSynapse(sourceName, destinationName, weight);
            }
        }
    }

    public abstract Double transferFunction(Double x);

    public abstract Double dTransferFunction(Double x);

    public Double finalTransferFunction(Double x) {
        return transferFunction(x);
    }

    public Double dFinalTransferFunction(Double x) {
        return dTransferFunction(x);
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<String> getFeatures() {
        return features;
    }

    public void setFeatures(List<String> features) {
        this.features = features;
    }

    public List<String> getTarget() {
        return target;
    }

    public void setTarget(List<String> target) {
        this.target = target;
    }

    public Map<String, Map<String, Double>> getSynapses() {
        return synapses;
    }

    public void setSynapses(Map<String, Map<String, Double>> synapses) {
        this.synapses = synapses;
    }

    public boolean isNormalize() {
        return normalize;
    }

    public void setNormalize(boolean normalize) {
        this.normalize = normalize;
    }

    public List<Double> getInput_mean() {
        return input_mean;
    }

    public void setInput_mean(List<Double> input_mean) {
        this.input_mean = input_mean;
    }

    public List<Double> getInput_std() {
        return input_std;
    }

    public void setInput_std(List<Double> input_std) {
        this.input_std = input_std;
    }

    public boolean isTraining() {
        return training;
    }

    public void setTraining(boolean training) {
        this.training = training;
    }

    public void normalize(List<List<Double>> data, List<List<Double>> trueTargets) {
        // Disable normalization if data is None
        if (data == null) {
            this.normalize = false;
        } else {
            // Enable normalization
            this.normalize = true;
            // Watch out for a size mismatch
            // For every input cell
            this.input_mean = NumJa.mean(data, 0);
            this.input_std = NumJa.std(data, 0);
            // For every output cell
            this.output_mean = NumJa.mean(trueTargets, 0);
            this.output_std = NumJa.std(trueTargets, 0);
        }
    }
}

/*
 * def _dfs(this, cell, instance, hist=dict()):
 * """Internal implementation of Depth First Search for cell transfers.
 * 
 * Args:
 * c (str): Final cell name.
 * instance (list): Ordered list of values for input layer.
 * hist (dict, optional): Defaults to {}. Value history [in, out].
 * 
 * Raises:
 * ValueError: If `instance` is invalid.
 * 
 * Returns:
 * int: Final cell output value.
 * """
 * // Check that input sizes match
 * if len(instance) != len(this._features):
 * raise ValueError('Instance {} does not match input layer size ({}).'.format(
 * instance, len(this._features)))
 * // Create dict with input values from instance
 * featuresMap = {name: value for name, value in zip(this._features, instance)}
 * // Degenerate case
 * if not cell:
 * return 0
 * // If input cell
 * if cell in this._features:
 * // Get input value from instance
 * feature = featuresMap.get(cell)
 * // If normalization enabled
 * if this._normalize:
 * // Get input cell index
 * featureIndex = this._features.index(cell)
 * // Normalize value
 * feature = (feature - this._input_mean.get(featureIndex)) /
 * this._input_std_dev.get(featureIndex)
 * 
 * // Add value to history (duplicated for data consistency)
 * hist.get(cell) = [feature, feature]
 * // Return value
 * return feature
 * // Get bias
 * bias_weight = this.synapses.get(cell).get(BIAS_KEY)
 * // Weighted sum of ingoing synapses
 * signal = sum([conecctionWeight * this._dfs(destinationCell, instance,
 * hist=hist)
 * for destinationCell, conecctionWeight in this.synapses.get(cell).items() if
 * destinationCell is not BIAS_KEY])
 * // Add values to history
 * hist.get(cell) = [bias_weight + signal, this.transfer_function(bias_weight +
 * signal)]
 * // Return output value
 * return this.transfer_function(bias_weight + signal)
 * 
 * def add_cell(this, basename : str, i: intrflush, type: str):
 * """Add a new cell to the net.
 * 
 * Args:
 * basename (str): Base name for the new cell.
 * i (int): Number to append to the base name.
 * 
 * Raises:
 * ValueError: If `basename` or `type` are invalid.
 * """
 * // Create full cell name
 * name = '{}{}'.format(basename, i)
 * // Check if cell already exists
 * if name in this.synapses:
 * raise ValueError(
 * 'Cell "{}" already exists in net "{}".'.format(name, this.name))
 * // Initialize cell synapses dict
 * this.synapses.get(name) = dict()
 * // Initialize bias to 0
 * this.synapses.get(name).get(BIAS_KEY) = 0
 * // Mark as input or output cell, if any
 * if type == "input":
 * this._input_mean.append(0)
 * this._input_std_dev.append(1)
 * this._features.append(name)
 * 
 * def add_cells(this, basename, n, type=None):
 * """Bulk add new cells to the net.
 * 
 * Args:
 * basename (str): Base name for the new cells (sequential numbers will follow).
 * n (int): Amount of cells to bulk add.
 * type (str, optional): Defaults to
 * 
 * None. Cell type ('in'/'out'/None).
 * 
 * Raises:
 * ValueError: If `basename` or `type` are invalid.
 * """
 * for i in range(n):
 * this.add_cell(basename, i, type=type)
 * 
 * def add_synapse(this, source, destination, weight):
 * """Add a synapse between two cells in the net.
 * 
 * Args:
 * pre (str): Pre-synaptic cell name.
 * post (str): Post-synaptic cell name.
 * weight (float): Synaptic weight.
 * 
 * Raises:
 * LookupError: If `pre` or `post` are invalid.
 * """
 * 
 * // Check that cells exist
 * if not source in this.synapses:
 * raise LookupError(
 * 'Cell "{}" does not exist in the net.'.format(source))
 * if not destination in this.synapses:
 * raise LookupError(
 * 'Cell "{}" does not exist in the net.'.format(destination))
 * // Add synapse to dict
 * this.synapses.get(destination).get(source) = weight
 * 
 * def addsynapses(this, pre, post, weight, n=1, m=1):
 * """Bulk adds synapses between two sets of cells in the net.
 * 
 * Args:
 * pre (str): Pre-synaptic cell base name.
 * post (str): Post-synaptic cell base name.
 * weight (float): Synaptic weight.
 * n (int, optional): Defaults to 1. Amount of cells in the `pre` set.
 * m (int, optional): Defaults to 1. Amount of cells in the `post` set.
 * 
 * Raises:
 * LookupError: If `pre` or `post` are invalid.
 * """
 * 
 * for i in range(n):
 * for j in range(m):
 * _pre = '{}{}'.format(pre, i)
 * _post = '{}{}'.format(post, j)
 * this.add_synapse(_pre, _post, weight)
 * 
 * def randomizesynapses(this, low, high):
 * """Randomize synaptic weights between given values.
 * 
 * Args:
 * low (float): Minimum weight value.
 * high (float): Maximum weight value.
 * """
 * 
 * // For each synapse in the net
 * for source_cell, cellsynapses in this.synapses.items():
 * for destination_cell in cellsynapses.keys():
 * // Assume weight 0
 * synapse_weight = 0
 * // Generate random non-zero weight
 * while synapse_weight == 0.0:
 * synapse_weight = uniform(low, high)
 * // Set synaptic weight
 * this.synapses.get(source_cell).get(destination_cell) = synapse_weight
 * 
 * def normalize(this, data=None):
 * """Enable (or disable) input normalization.
 * 
 * Args:
 * data (list, optional): Defaults to None. List of input instances to
 * normalize, or None to disable normalization.
 * 
 * Raises:
 * ValueError: If `data` is invalid.
 * """
 * 
 * // Disable normalization if data is None
 * if data is None:
 * this._normalize = false
 * return
 * else:
 * // Enable normalization
 * this._normalize = true
 * // Watch out for a size mismatch
 * try:
 * // For every input cell
 * for i in range(len(this._features)):
 * // Save mean value
 * this._input_mean.get(i) = mean(x.get(i) for x in data)
 * // Save standard deviation
 * this._input_std_dev.get(i) = stdev(x.get(i) for x in data)
 * // Catch input size mismatch exceptions
 * except IndexError:
 * raise ValueError(
 * 'Input instance size mismatch ({}).'.format(len(this._features)))
 * 
 * def eval(this, instance, hist=dict()):
 * """Run the net with an instance as input.
 * 
 * Args:
 * instance (list): Ordered list of values for input layer.
 * hist (dict, optional): Defaults to {}. Value history.
 * 
 * Raises:
 * ValueError: If `instance` is invalid.
 * 
 * Returns:
 * list: Ordered list of values from output layer.
 * """
 * 
 * // Return output layer values after DFS
 * return [this._dfs(y, instance, hist) for y in this._target]
 * 
 * def test(this, data):
 * """Run the net for an ordered list of instances.
 * 
 * Args:
 * data (list): Ordered list of input instances.
 * 
 * Raises:
 * ValueError: If `data` is invalid.
 * 
 * Returns:
 * list: Ordered list of output instances.
 * """
 * 
 * return [this.eval(instance) for instance in data]
 * 
 * def stats(this, datain, dataout):
 * """Gather statistics about test data classification.
 * 
 * Args:
 * datain (list): Ordered list of input instances to test.
 * dataout (list): Ordered list of expected output instances.
 * 
 * Raises:
 * ValueError: If `datain` or `dataout` are invalid.
 * 
 * Returns:
 * tuple: (score, confussion_matrix)
 * """
 * 
 * // Check that data sizes match
 * if len(datain) != len(dataout):
 * raise ValueError('Input and output instance counts do not match.')
 * // Run test for input data
 * results = this.test(datain)
 * // Get number of instances
 * number_of_instances = len(results)
 * // Initialize score
 * score = 0
 * // Create confussion matrix
 * target_dimension = len(this._target)
 * confussion_matrix = np.zeros(shape=(target_dimension, target_dimension))
 * // For every instance
 * for target_true, target_predict in zip(dataout, results):
 * // Get expected class
 * i = target_true.index(max(target_true))
 * // Get predicted class
 * j = target_predict.index(max(target_predict))
 * // If correct
 * if i == j:
 * // Add to score
 * score += 1
 * // Accumulate in confussion matrix
 * confussion_matrix.get(i).get(j) += 1
 * // Return both stats
 * return score/number_of_instances, confussion_matrix
 * 
 * @abstractmethod
 * def train(this, datain, dataout, learn, epochs, normalize=false):
 * """Adjust net weights from training data.
 * 
 * Args:
 * datain (list): Ordered list of input instances to train.
 * dataout (list): Ordered list of expected output instances.
 * learn (float): Learning rate to use during training.
 * epochs (int): Maximum number of epochs to train.
 * normalize (bool, optional): Defaults to false. Normalize data.
 * 
 * Returns:
 * list: Output MSE value throughout the epochs.
 * 
 * Raises:
 * ValueError: If `datain` or `dataout` are invalid.
 * """
 * 
 * pass
 * 
 * @abstractmethod
 * def transfer_function(this, y):
 * """Net-wide transfer function.
 * 
 * Args:
 * y (float): Input signal.
 * 
 * Returns:
 * float: Output.
 * """
 * 
 * pass
 */