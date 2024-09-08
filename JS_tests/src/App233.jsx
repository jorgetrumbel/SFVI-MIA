import { useState, useEffect } from "react";
import TaskList from "./components/TaskList";
import TaskForm from "./components/TaskForm";
import { tasks as data } from "./tasks";

function App() {
  const [count, setCount] = useState(0);
  const [taskList, setTaskList] = useState([]);

  useEffect(() => {
    setTaskList(data);
  }, []);

  function createTask(task) {
    setTaskList([...taskList, {
      id: taskList.length+1,
      title: task.title,
      description: task.description
    }]);
    console.log(taskList)
  }

  function deleteTask(id){
    setTaskList(taskList.filter(task => task.id !== id))
  }

  return (
    <>
      <TaskForm createTask={createTask} />
      <TaskList taskList={taskList} deleteTask={deleteTask}/>
    </>
  );
}

export default App;
