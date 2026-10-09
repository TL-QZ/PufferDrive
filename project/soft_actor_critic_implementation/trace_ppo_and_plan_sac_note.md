# 2.2, 3.2
- need more detailed design understanding for experience buffer
- SAC vs PPO experience reuse difference
- SAC can use more historical experience, some portion of the experience can be kept while other can be resampled from behavior policy or current target policy
- what is ring buffer? Why is it used in for SAC?
- sac seems to also train on 1-step transitions, so does it have to wait until episode terminates?
- how similar is SAC experience to PPO experience?

# 3.2
- so what need to be changed from PPO collection?

# 3.4
- make sure do so cleanly and storage efficent but also fast, make sure it is the best most intuitive way

# 3.5
- needs to check [planning.md, replay collection](planning.md#4-replay-collection-is-the-first-correctness-gate)
- TODO: dig deeper in to this turuncation and asynchronous experience handling
- do not understand why current handling is not sufficient for SAC (lack of knowledge of the PPO and SAC algo specifics)

# 4.8
- recurrent PPO adaption to SAC is not top priority


---

# reponse to agent

1. regarding point 1

if values and logprobs not necesary for SAC, then omit them, do not create that self.values and self.logprobs during initialization, is that right?

action can comefrom the behavior policy so that is no problem

2. regarding point 2

so what is then plan to do for this point about this "shift". so do you mean when we implement say q value estimation, we need to account for this shift in the rewards? (if so then that is just the correct thing to do, and it is not we are doing any major shift and the addition of function is for action value estimation or other purpose not explicitly for compensate this shift)

so the r come back from vecenv.recv is for that state, which is either the initial state or state reached due to the previous action.

3. regarding point 3

this design of frequence to update the experience buffer is reasonable mechinism to add, but it does not stop us to doing it in a really clean, efficient and intuitive way.

so are you referring if the for loop end it ends at self.vecenv.send(action)? that should not be a hard case to handle right? like we just do not do that at the end of for loop or just exclude the last step in some way?

4. regarding point 4

and handling this issue seems just to be using the termination flag correctly?

5. regarding point 5

I see no issue just mirror how PufferRL.evaluate() is implemented and do modifications accordingly. current PufferSAC.evaluate() is not doing that, it is having self.collector as extra layer of complexity. yet it is not fully implemented, but i do not see why we have to use it and not faithfully follow PufferRL.evaluate() with appropriate modification while maintaining efficiency.