package com.example.nguyen.practicemachine;


import android.os.AsyncTask;
import android.util.Log;
import android.widget.EditText;


import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Nguyen on 2/16/2017.
 */

public class NeuralNetwork extends AsyncTask<Integer,Integer,Integer> {

    protected Integer doInBackground(Integer... params)
    {
        int count = params.length;
        createAndUseNetwork(params[0],params[1]);
        return null;
    }

    public void createAndUseNetwork(int param1, int param2)
    {
        Log.d("parmas", param1 + " " +param2);
        //Create our neural network layers and node in/out
        DenseLayer inputLayer = new DenseLayer.Builder()
                .nIn(2)
                .nOut(2)
                .name("Input")
                .build();
        DenseLayer hiddenLayer = new DenseLayer.Builder()
                .nIn(2)
                .nOut(2)
                .name("Hidden")
                .build();

        OutputLayer outputLayer = new OutputLayer.Builder()
                .nIn(2)
                .nOut(2)
                .name("Output")
                .activation(Activation.SOFTMAX)
                .build();

        //Set our iterations of configuration as well as learning rate
        NeuralNetConfiguration.Builder nncBuilder = new NeuralNetConfiguration.Builder();
        nncBuilder.iterations(60000);
        nncBuilder.learningRate(0.1);

        //Set each layer to a number in a list
        NeuralNetConfiguration.ListBuilder listBuilder = nncBuilder.list();
        listBuilder.layer(0,inputLayer);
        listBuilder.layer(1,hiddenLayer);
        listBuilder.layer(2,outputLayer);


        //Set parameters for backpropogation
        listBuilder.backprop(true);

        MultiLayerNetwork myNetwork = new MultiLayerNetwork(listBuilder.build());
        myNetwork.init();

        final int NUM_SAMPLES = 20;


        INDArray trainingInputs = Nd4j.zeros(NUM_SAMPLES, inputLayer.getNIn());
        INDArray trainingOutputs = Nd4j.zeros(NUM_SAMPLES, outputLayer.getNOut());
// If 0,0 show 0
        trainingInputs.putScalar(new int[]{0,0}, 0);
        trainingInputs.putScalar(new int[]{0,1}, 0);
        trainingOutputs.putScalar(new int[]{0,0}, 0);
        //trainingOutputs.putScalar(new int[]{0,1}, 1);

// If 0,1 show 1
        trainingInputs.putScalar(new int[]{1,0}, 0);
        trainingInputs.putScalar(new int[]{1,1}, 1);
        trainingOutputs.putScalar(new int[]{1,0}, 1);
        //trainingOutputs.putScalar(new int[]{1,1}, 0);

// If 1,0 show 1
        trainingInputs.putScalar(new int[]{2,0}, 1);
        trainingInputs.putScalar(new int[]{2,1}, 0);
        trainingOutputs.putScalar(new int[]{2,0}, 1);
        //trainingOutputs.putScalar(new int[]{2,1}, 0);

// If 1,1 show 0
        trainingInputs.putScalar(new int[]{3,0}, 1);
        trainingInputs.putScalar(new int[]{3,1}, 1);
        trainingOutputs.putScalar(new int[]{3,0}, 0);
        //trainingOutputs.putScalar(new int[]{3,1}, 1);
        //Set up dataset

        DataSet myData = new DataSet(trainingInputs,trainingOutputs);

        myNetwork.fit(myData);

        INDArray actualInput = Nd4j.zeros(1,2);
        actualInput.putScalar(new int[]{0,0}, 1);
        actualInput.putScalar(new int[]{0,1}, 1);
        Log.d("inputs", actualInput.toString());
        //Generate output
        INDArray actualOutput = myNetwork.output(actualInput);
        Log.d("Output", actualOutput.toString());


    }
}
