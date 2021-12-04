using System;
using System.Linq;
using System.Collections.Generic;
namespace AgentTemplate {
    public interface IAgent<TInput, TOutput> {
        TOutput StepObserveRespond(TInput observation);
        void AddReward(float reward);
        void EndEpisode();
    }
    public interface ITrainableAgent<TInput, TOutput> : IAgent<TInput, TOutput> {
        IStep<TInput, TOutput>[] GetTakenSteps();
    }
    public interface IStep<TInput, TOutput> {
        float GetReward();
        bool GetIsTerminal();
        TInput GetObservation();
        TOutput GetResponse();
    }
    public class Agent<TInput, TOutput> : IAgent<TInput, TOutput> {
        private readonly Func<TInput, TOutput> policy;
        /// <summary>
        /// Initialize an agent with a function that takes parameter type TInput and returns type TOutput
        /// </summary>
        /// <param name="policy"></param>
        public Agent(Func<TInput, TOutput> policy) {
            this.policy = policy;
        }
        /// <summary>
        /// Increment step and get agent's response to observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public TOutput StepObserveRespond(TInput observation) {
            TOutput response = policy.Invoke(observation);
            return response;
        }
        /// <summary>
        /// Does nothing since this is not a trainable agent
        /// </summary>
        /// <param name="reward"></param>
        public void AddReward(float reward) { }
        /// <summary>
        /// Does nothing since this is not a trainable agent
        /// </summary>
        public void EndEpisode() { }
    }
    public class AgentExtender<TInput, TOutput> : IAgent<TInput, TOutput> {
        private readonly IAgent<TInput, TOutput> agent;
        public AgentExtender(IAgent<TInput, TOutput> agent) {
            this.agent = agent;
        }
        public virtual void AddReward(float reward) {
            agent.AddReward(reward);
        }
        public virtual void EndEpisode() {
            agent.EndEpisode();
        }
        public virtual TOutput StepObserveRespond(TInput observation) {
            return agent.StepObserveRespond(observation);
        }
    }
    public class TrainableAgent<TInput, TOutput> : AgentExtender<TInput, TOutput>, ITrainableAgent<TInput, TOutput> {
        private List<Step> steps = new List<Step>();
        private int stepsBufferSize;
        /// <summary>
        /// Initialize an agent that is use for training
        /// </summary>
        /// <param name="agent">The agent to be train</param>
        /// <param name="stepsBufferSize">The maximum amount of steps to keep in memory</param>
        public TrainableAgent(IAgent<TInput, TOutput> agent, int stepsBufferSize = 128) : base(agent) {
            this.stepsBufferSize = stepsBufferSize;
        }
        /// <summary>
        /// Adds the previous step into memory that can be use for training. Increment step and get agent's response to observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public override TOutput StepObserveRespond(TInput observation) {
            TOutput response = base.StepObserveRespond(observation);
            steps.Add(new Step(observation, response));
            while (steps.Count > stepsBufferSize) {
                steps.RemoveAt(0);
            }
            return response;
        }
        /// <summary>
        /// Add reward to current step
        /// </summary>
        /// <param name="reward"></param>
        public override void AddReward(float reward) {
            base.AddReward(reward);
            if (steps.Count != 0) {
                steps.Last().AddReward(reward);
            }
        }
        /// <summary>
        /// Ends episode
        /// </summary>
        public override void EndEpisode() {
            base.EndEpisode();
            if (steps.Count != 0) {
                steps.Last().EndEpisode();
            }
        }
        /// <summary>
        /// Returns the steps in memory
        /// </summary>
        /// <returns></returns>
        public IStep<TInput, TOutput>[] GetTakenSteps() {
            List<IStep<TInput, TOutput>> _steps = new List<IStep<TInput, TOutput>>(steps);
            _steps.RemoveAt(_steps.Count - 1);
            return _steps.ToArray();
        }
        /// <summary>
        /// Contains the step's information
        /// </summary>
        private class Step : IStep<TInput, TOutput> {
            private TInput observation;
            private TOutput response;
            private float reward;
            private bool isTerminal;
            public Step(TInput observation, TOutput response) {
                this.observation = observation;
                this.response = response;
            }
            internal void AddReward(float reward) {
                if (isTerminal == false) {
                    this.reward += reward;
                }
            }
            internal void EndEpisode() {
                isTerminal = true;
            }
            /// <summary>
            /// Was this step terminal (The step before the episode ended)?
            /// </summary>
            /// <returns></returns>
            public bool GetIsTerminal() {
                return isTerminal;
            }
            /// <summary>
            /// Returns agent's observation for this step
            /// </summary>
            /// <returns></returns>
            public TInput GetObservation() {
                return observation;
            }
            /// <summary>
            /// Returns agent's response for this step
            /// </summary>
            /// <returns></returns>
            public TOutput GetResponse() {
                return response;
            }
            /// <summary>
            /// Returns total rewards gained during this step
            /// </summary>
            /// <returns></returns>
            public float GetReward() {
                return reward;
            }
        }
    }
}