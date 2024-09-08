import TaskItem from "./TaskItem";


function TaskList({taskList, deleteTask}) {

  if (taskList.length === 0) {
    return <h1>No tasks</h1>;
  }

  return (
    <div>
      {taskList.map((task, index) => (
        <TaskItem key={index} task={task} deleteTask={deleteTask} />
      ))}
    </div>
  );
}

export default TaskList;
