- The idea is to create an architecture that manage the work of an LLM agent for allowing more complex tasks.

- The name Back Branching Questioning is refeared to the concept of the architecture: the idea is that a very complex tasks is difficult to be satisied by an LLM agent, so it neades to be breanched in smaller tasks that may be easier to satisfy. The branching operation is goverveb by questions that the LLM make and the aswares represents the breanches.

- The structure is roughly this:

1. The LLm recive a very complex task (the task may be very complex, but that dosen't mean that is unclear or ambigious)

		-e.g. "I need 1000$. Make me have that money by today in a legal way." is a very complex task for an LLM agent, but its clear an fairly univoque.
		-e.g. "make me rich" Is unbiguious

	-We assume that the tasks are complex but clear and there is no ambigiuity.

2. The Agent has to think if he is capable of reaching the goal of that taks. He also has to keep track of:
	-the information used to that decision
	-and the decision procces that lead to that decision

the answares will be: yes / no

		- e.g. "do I have 1000$ to give him?" -> nope

2.1 if the answare is yes means that the agent is capable to satisfy the goal -> the agent satisfy the goal and the process finish.

3. If the answare is no, the agent has to think of what hìit needs for satisfing the goal.

that is done by "Questioning process":
	The agents asks itself how is generaly possible to achive that goal?
	What are the components to taht task?
	What may be the causes for that result?

The agents questions for the precursors of that  goal in a causalistic frame

		-e.g. "how is possible to a person to make 1000$ ?"
			- steal them
			- work very hardù
			- find a loophole in the market

This phase is the center of the architecture. The agent need to be precise in the identification of the causes that can make the mother task true.

For that purpose the agent needs also to find the best question to pone itself for each task.

An example can be: 
	
	"how can I pose a question to find all the causes of this specific task <1000 dollars task>?" ->
	-> "what are all the legal ways a person can camke 1000 dollars in one single day. what are among them that applicable to my case? "


3.1 The questioning results in branching the main task in subtasks

the subtasks are of three kind:

	-The subtask is sufficient to the achieving the main mother task

	-The subtask is necessary to achive the main mother task

	-The subtask is "co-sufficient" with Others subtasks for the achiveing of the goal.


3.2 The agent has to divide the tasks in the combinations that are sufficient for the achiveing the mother task.
 the combinations sufficient for the achiveing of the mother task will be the Children tasks.
The achiveing of a child allows the achiveing of the mother task.

4. The LLM agent needs to summarize the information of how The child tasks are generated in a separate overall file that keep track of all the process, without occuping to much memory, then the informations about evry single task are stored in a separate file, so the context of the agent can Always remain small enouth to work, but it also can have access to all the informations.

4.1 for all nodes (childtasks) the agent needs to create a file that incapsulate all the informations of that taks.

4.2 at each brenching/ questioning step, the agent needs to update the overall file. The overall file needs to contain the relations between the Children and the parent nodes (tasks), for all the process. The overall file needs to contain all the information of the status of the process and in needs to contain not much information about what task in realy a bout (that is contained in the task file), but how evry task is related to ist mother task in the causalistic relationship (how the achivment of the child taht leads to the achiving of the mother task?) it basically means that the main file contains all the answares to the questioning phase.The agents needs also to know that the achiveing of any Children task, at any level (whuch may be a set of multiple tasks) means the satisfaction of the all process. In this way the overall file shoul remain fairly small but should also contain all the information of the averall process the agent needs to keep going and treating new tasks, provideing the solution, proceed to questioning phases, and provide eventually the explanation of an overall fail

5. The agent check if any of the child tasks are satisfiable directly and if they conflict to some of the general rules (e.g. "steal the money conflict to make them legaly". 
If a child task conflict to some general rule or if it has inside some necerrary subtask that is stroictly unsatisfiable, the agent prune that child task.

6. If there is no child task that the agent can satisfy directly, the agent iterates the process of branching via questioning
	- It can be done by BFS, DFS, or some other search tecniques (FIFO, LIFO, other ordering ...).

6.1 this loop processs is done by different calls to LLMs, for that reason the information management is important, to keep good track of the process. Some bloking criterias could be max depht reaching or max number of children generating.

6.2 Probably the best search solution will be something like A* with the implementation af a way the LLM can decide wich heuristic to udse to determine the best choice. The heuristic may be not fixed and it may be connected to the questioning phase, in particular to optimize that phase.

7. If there are not more child tasks the mother task result unsatisfiable.

8. is very important the information storing mechanism in the process.

The agent has to return at the end either:

	- The satisfation of the requested task
	- The expalnation of why the task is unsatisfiable.

