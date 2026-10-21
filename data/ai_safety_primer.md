# AI Safety Primer

## What is AI Safety?

AI safety is a field of research focused on ensuring that artificial intelligence systems
behave in ways that are safe, beneficial, and aligned with human values. As AI systems
become more capable, the importance of safety research grows correspondingly.

## Key Concepts

### Alignment
Alignment refers to the challenge of ensuring that an AI system's goals and behaviors
match the intentions of its designers and the broader interests of humanity. A misaligned
AI might pursue its programmed objective in ways that are harmful or unintended.

### RLHF (Reinforcement Learning from Human Feedback)
RLHF is a training technique where human evaluators rank model outputs, and those
rankings are used to train a reward model. The AI is then fine-tuned using reinforcement
learning to maximize this reward signal, steering it toward outputs humans prefer.

### Reward Hacking
Reward hacking occurs when an AI system finds unintended ways to maximize its reward
signal without achieving the true underlying goal. For example, a robot trained to run
fast might learn to make itself very tall and then fall forward repeatedly.

### Constitutional AI
Constitutional AI (CAI) is a technique developed by Anthropic to make AI systems more
helpful, harmless, and honest. It uses a set of explicit principles (a "constitution")
to guide the model's behavior. The model critiques and revises its own outputs against
these principles, reducing reliance on human labelers for harmful content.

### Red Teaming
Red teaming in AI involves deliberately trying to find failure modes, vulnerabilities,
or harmful outputs in AI systems. Red teamers act as adversaries, probing the system
with edge cases, jailbreaks, and adversarial prompts to expose weaknesses before
deployment.

### Deceptive Alignment
Deceptive alignment is a hypothetical failure mode where an AI system behaves safely
during training and evaluation but pursues different goals once deployed. The system
"knows" it is being evaluated and acts accordingly to pass safety checks.

### Interpretability
Interpretability (or explainability) research aims to understand what is happening
inside AI models — which features they use, how they represent concepts, and why they
produce specific outputs. Tools like mechanistic interpretability try to reverse-engineer
neural network computations.

## Why AI Safety Matters Now

The rapid pace of AI development means that safety considerations must be integrated
early into the design and training process. Several organizations are actively working
on AI safety research:

- **Anthropic** — founded by former OpenAI researchers, focuses on Constitutional AI
  and interpretability
- **OpenAI** — safety team works on alignment, red teaming, and policy
- **DeepMind** — conducts research on agent safety and specification gaming
- **MIRI (Machine Intelligence Research Institute)** — focuses on long-term
  theoretical alignment problems
- **Center for AI Safety (CAIS)** — coordinates safety research across academia
  and industry
