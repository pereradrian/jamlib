package com.oddkernel.jamblib;

import java.util.List;

import javax.swing.JFrame;

import org.jfree.chart.ChartColor;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class ScatterPlot extends JFrame {
    private static final long serialVersionUID = 1L;
	private JFreeChart chart;  

    public ScatterPlot(String name, List<List<Double>> trainFeatures, List<List<Double>> trainTargets, List<List<Double>> trueTargets) { 
        super(name);

        // Create dataset  
        XYDataset dataset = createDataset(trainFeatures, trainTargets, trueTargets);  
    
        // Create chart  
        this.chart = ChartFactory.createScatterPlot("Sample data", "X-Axis", "Y-Axis", dataset, PlotOrientation.VERTICAL, true, false, false);

        //Changes background color  
        XYPlot plot = (XYPlot)chart.getPlot();  
        plot.setBackgroundPaint(new ChartColor(255,228,196));  
        
    }


    private XYDataset createDataset(List<List<Double>> features, List<List<Double>> targets, List<List<Double>> trueTargets) {
        XYSeriesCollection dataset = new XYSeriesCollection();  
        XYSeries perceptronSeries = new XYSeries("Perceptron");
        XYSeries polynomialSeries = new XYSeries("Polynomial");
        for (int indexSample = 0; indexSample < features.size(); indexSample++) {
            Double x = features.get(indexSample).get(0);
            Double yPredict = targets.get(indexSample).get(0); 
            perceptronSeries.add(x, yPredict); 
            if (trueTargets != null) {   
                Double yTrue = trueTargets.get(indexSample).get(0);
                polynomialSeries.add(x, yTrue);
            }
        }
        dataset.addSeries(perceptronSeries);
        if (trueTargets != null) {   
            dataset.addSeries(polynomialSeries);
        }
      
        return dataset;  
	}


	public JFreeChart getChart() {
		return chart;
	}

}
