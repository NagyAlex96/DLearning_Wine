using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DLearning_Wine
{
    public class WineModel
    {
        private int hiddenLayerDim = 100; // Megnövelt rejtett réteg méret a jobb pontosságért
        public Function Model { get; private set; }

        // Modell létrehozása
        public WineModel(Variable input, int outputDim, DeviceDescriptor device)
        {
            var layer1 = FullyConnectedLayer(input, hiddenLayerDim, device, Activation.ReLU);
            Model = FullyConnectedLayer(layer1, outputDim, device, Activation.None);
        }

        // Rejtett réteg létrehozása adott aktivációs függvénnyel
        private Function FullyConnectedLayer(Variable input, int outputDim, DeviceDescriptor device, Activation activation)
        {
            var inputDim = input.Shape[0];
            var weights = new Parameter(new int[] { outputDim, inputDim }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device);
            var bias = new Parameter(new NDShape(1, outputDim), DataType.Float, 0.0f, device);
            var fullyConnected = CNTKLib.Plus(bias, CNTKLib.Times(weights, input));

            switch (activation)
            {
                case Activation.ReLU:
                    return CNTKLib.ReLU(fullyConnected);
                case Activation.Sigmoid:
                    return CNTKLib.Sigmoid(fullyConnected);
                default:
                    return fullyConnected;
            }
        }

        // Előrejelzések készítése
        public int Predict(float[] features, DeviceDescriptor device)
        {
            var inputVal = Value.CreateBatch(Model.Arguments[0].Shape, features.ToList(), device);
            var outputDataMap = new Dictionary<Variable, Value> { { Model.Output, null } };
            Model.Evaluate(new Dictionary<Variable, Value> { { Model.Arguments[0], inputVal } }, outputDataMap, device);

            var outputData = outputDataMap[Model.Output].GetDenseData<float>(Model.Output);
            return outputData[0].IndexOf(outputData[0].Max());
        }
    }
}
