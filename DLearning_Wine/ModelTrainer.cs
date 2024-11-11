using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DLearning_Wine
{
    public class ModelTrainer
    {
        private int epochSize;
        private int minibatchSize;
        private DeviceDescriptor device;
        private Trainer trainer;

        // Konstruktor, amely a tanítást és a tanulási paramétereket állítja be
        public ModelTrainer(Function model, Variable label, int epochSize, int minibatchSize, DeviceDescriptor device)
        {
            this.epochSize = epochSize;
            this.minibatchSize = minibatchSize;
            this.device = device;
            var loss = CNTKLib.CrossEntropyWithSoftmax(model, label);
            var evalError = CNTKLib.ClassificationError(model, label);
            var learner = CNTKLib.SGDLearner(new ParameterVector(model.Parameters().ToArray()), new TrainingParameterScheduleDouble(0.01, 1));
            trainer = Trainer.CreateTrainer(model, loss, evalError, new List<Learner> { learner });
        }

        // Modell edzése az előkészített adatokkal
        public void TrainModel(Variable input, Variable label, List<Tuple<float[], float[]>> trainingData)
        {
            for (int epoch = 0; epoch < epochSize; epoch++)
            {
                var shuffledData = trainingData.OrderBy(_ => Guid.NewGuid()).ToList();
                for (int i = 0; i < shuffledData.Count; i += minibatchSize)
                {
                    var minibatchData = shuffledData.Skip(i).Take(minibatchSize).ToList();
                    var featuresBatch = minibatchData.Select(d => d.Item1).ToArray();
                    var labelsBatch = minibatchData.Select(d => d.Item2).ToArray();

                    var inputVal = Value.CreateBatch(input.Shape, featuresBatch.SelectMany(f => f).ToList(), device);
                    var labelVal = Value.CreateBatch(label.Shape, labelsBatch.SelectMany(l => l).ToList(), device);

                    var dataMap = new Dictionary<Variable, Value> { { input, inputVal }, { label, labelVal } };
                    trainer.TrainMinibatch(dataMap, device);
                }
            }
        }

        // Modell pontosságának kiértékelése
        public float EvaluateModel(WineModel model, List<Tuple<float[], float[]>> testData)
        {
            int correct = 0;

            foreach (var sample in testData)
            {
                if (sample.Item1 == null || sample.Item1.Length == 0) continue;

                var predictedLabel = model.Predict(sample.Item1, device);
                var actualLabel = Array.IndexOf(sample.Item2, 1.0f);

                if (predictedLabel == actualLabel) correct++;
            }

            return correct / (float)testData.Count * 100;
        }
    }
}
