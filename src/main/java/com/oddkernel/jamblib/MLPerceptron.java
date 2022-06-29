package com.oddkernel.jamblib;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.jfree.chart.ChartUtils;

public class MLPerceptron extends Net {

    private static final String NAME_H = "z";
    private static final String NAME_O = "y";
    private static final String NAME_I = "x";
    private static final int IN = 0;
    private static final int OUT = 1;
	private static final double EPS = 1e-16;

    private int featureDimension;
    private int targetDimension;
    private List<Integer> hiddenLayerSizes;
    private List<String> hiddenLayerNames;
    private boolean training = false;

    public MLPerceptron(String name, int featureDimension, int targetDimension, List<Integer> hiddenLayerSizes) {
        super(name);
        this.hiddenLayerSizes = hiddenLayerSizes;
        if (this.hiddenLayerSizes == null) {
            throw new IllegalArgumentException("Invalid hidden layers sizes.");
        }
        this.featureDimension = featureDimension;
        this.targetDimension = targetDimension;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.hiddenLayerNames = new ArrayList<>();

        // List<String> hiddenLayerSizesString = hiddenLayerSizes.stream().map(value -> String.valueOf(value)).collect(Collectors.toList());

        // For each hidden layer
        for (int indexLayer = 0; indexLayer < hiddenLayerSizes.size(); indexLayer++) {

            // Create base name
            String layerName = NAME_H + "_" + String.valueOf(indexLayer + 1);
            this.hiddenLayerNames.add(layerName);

            // Add hidden cells
            int layerSize = hiddenLayerSizes.get(indexLayer);
            this.addCells(layerName, layerSize);
            // Add synapses with previous layer

            if (0 < indexLayer) {
                String previousLayerName = hiddenLayerNames.get(hiddenLayerNames.size() - 2);
                this.addSynapses(previousLayerName, layerName,
                        0.0, hiddenLayerSizes.get(indexLayer - 1), hiddenLayerSizes.get(indexLayer));
            }
        }
        // Add hidden cells
        // Add input layer cells
        this.addCell(NAME_I, featureDimension, "input");
        // Add output layer cells
        this.addCell(NAME_O, targetDimension, "output");

        // Add input synapses
        this.addSynapses(NAME_I, this.hiddenLayerNames.get(0), 0.0, featureDimension, hiddenLayerSizes.get(0));
        // Add output synapses
        int indexLayer = hiddenLayerSizes.size() - 1;
        this.addSynapses(this.hiddenLayerNames.get(indexLayer), NAME_O, 0.0, hiddenLayerSizes.get(indexLayer),
                targetDimension);
    }

    public List<Double> train(List<List<Double>> features, List<List<Double>> trueTargets, double learning_rate,
            int epochs, boolean normalize) {
        this.training = true;
        if (features.size() != trueTargets.size()) {
            throw new IllegalArgumentException("Input and output instance counts do not match.");
        }
        if (learning_rate <= 0.0 || 1.0 < learning_rate) {
            throw new IllegalArgumentException("Learning rate must be within the interval (0, 1].");

        }
        System.out.println("Training... 0%");
        if (normalize) {
            this.normalize(features, trueTargets);
        }

        // Initiallize
        boolean stop = false;
        List<Double> mse = new ArrayList<>();
        int epoch = 0;
        int sampleSize = features.size();

        // Run epochs until stop conditions are met
        while (!stop && epoch < epochs) {
            // Assume stop condition
            stop = true;
            for (int indexFeature = 0; indexFeature < sampleSize; indexFeature++) {
                List<Double> feature = features.get(indexFeature);
                List<Double> trueTarget = trueTargets.get(indexFeature);
                // Use auxiliary variables
                Map<String, Map<String, Double>> synapses = new HashMap<>(this.getSynapses());
                // Populate history dict and get output values
                Map<String, List<Double>> historial = new HashMap<>();
                List<Double> predictTarget = this.eval(feature, historial);
                List<Double> delta_in = NumJa.minus(trueTarget, predictTarget);

                // Create lists for retropropagation
                List<String> names = getNames();
                List<Integer> sizes = getSizes();
                // Retropropagation
                for (int k : NumJa.arange(names.size()-1)) {
                    // Initialize list for previous layer _delta_in
                    List<Double> prev_delta_in = NumJa.zeros(sizes.get(k + 1));
                    Integer inputSize = sizes.get(k);
                    for (int j : NumJa.arange(inputSize)) {
                        // If correction is needed
                        if (EPS < Math.abs(delta_in.get(j))) {
                            String _zz = String.format("%s_%s", names.get(k), j+1);
                            // Get input value from history
                            Double zz_in = historial.get(_zz).get(IN);
                            double delta = 0.0;
                            if (k==0) {
                                delta = delta_in.get(j) * this.dFinalTransferFunction(zz_in);
                            } else {

                                delta = delta_in.get(j) * this.dTransferFunction(zz_in);
                            }

                            // Save bias weight correction
                            Double currentUpdate = synapses.get(_zz).get(BIAS_KEY);
                            Double newValue = currentUpdate + learning_rate * delta;
                            synapses.get(_zz).put(BIAS_KEY, newValue);

                            // For each cell in previous layer
                            for (int i : NumJa.arange(sizes.get(k+1))) {
                                String _z = String.format("%s_%s", names.get(k + 1), i + 1);
                                // Get output value from history
                                Double z = historial.get(_z).get(OUT);
                                // Save synaptic weight correction
                                currentUpdate = synapses.get(_zz).get(_z);
                                newValue = currentUpdate + learning_rate * delta * z;
                                synapses.get(_zz).put(_z, newValue);
                                // Accumulate delta_out
                                currentUpdate = prev_delta_in.get(i);
                                prev_delta_in.set(i, currentUpdate + delta * newValue);
                            }
                            stop = false;
                        }
                    }
                    // Set input deltas for previous layer
                    delta_in = new ArrayList<>(prev_delta_in);
                }
                // Update synapses
                this.setSynapses(synapses);
            }
            // Test dataset at the end of current epoch
            List<List<Double>> predictTargets = this.test(features);
            /**
             * 
            SwingUtilities.invokeLater(() -> {  
                plot.setSize(800, 400);  
                plot.setLocationRelativeTo(null);  
                plot.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);  
                plot.setVisible(true);  
            });
            */
            if (epoch % 10 == 0) {
                ScatterPlot plot = new ScatterPlot("Function fit", features, predictTargets, trueTargets);
                try {
                    ChartUtils.saveChartAsPNG(new File(String.format("./img/poly_predict_%05d.png", epoch+1)), plot.getChart(), 450, 400);
                }
                catch (IOException io) {
                    System.out.println("ERROR SAVING PICTURE");
                }
            }


            // Compute residual sum of squares
            Double error = NumJa.mse(predictTargets, trueTargets);
            // Append MSE value after current epoch
            mse.add(error);
            // Move to next epoch
            epoch++;
            // System.out.println current progress
            System.out.println(String.format("Training... %.2f%% mse: %.4f %s %s", 100.0 * epoch / epochs, error, !stop, epoch < epochs));
        }

        this.training = false;
        // Return MSE list
        return mse;
    }

    private List<List<Double>> test(List<List<Double>> features) {
        HashMap<String, List<Double>> historial = new HashMap<>();
        return features.stream().map(feature -> this.eval(feature, historial)).collect(Collectors.toList());
    }

    private List<Integer> getSizes() {
        List<Integer> result = new ArrayList<>();
        result.add(targetDimension);
        for (int indexName = this.hiddenLayerNames.size() -1; 0 <= indexName; indexName--) {
            result.add(this.hiddenLayerSizes.get(indexName));
        }
        result.add(featureDimension);
        return result;
    }

    private List<String> getNames() {
        List<String> result = new ArrayList<>();
        result.add(NAME_O);
        for (int indexName = this.hiddenLayerNames.size()-1; 0 <= indexName; indexName--) {
            result.add(this.hiddenLayerNames.get(indexName));
        }
        result.add(NAME_I);
        return result;
    }

    @Override
    public Double transferFunction(Double x) {
        // Watch out for overflows
        if (Math.abs(x) > 10) {
            if (x < 0.0) {
                return -1.0;
            } else {
                return 1.0;
            }
        } else {
            return 2.0 / (1.0 + NumJa.exp(-x)) - 1.0;
        }
    }

    @Override
    public Double dTransferFunction(Double x) {
        return (1.0 + this.transferFunction(x)) * (1.0 - this.transferFunction(x)) / 2.0;
    }
}
