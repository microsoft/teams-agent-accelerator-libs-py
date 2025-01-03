// import Breadcrumb from '../components/Breadcrumbs/Breadcrumb';
import DataTable from 'react-data-table-component';
import { useForm } from 'react-hook-form';


const SearchBar = () => {
  const { register, handleSubmit, formState: { errors } } = useForm();
  const onSubmit = (data: any) => console.log(data);
  console.log(errors);

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4 p-4 bg-gray-200 rounded-md">
      <div className="flex gap-4">
        <div className="flex flex-col flex-1">
          <label className="mb-1 text-gray-700">User Id</label>
          <input
            type="text"
            placeholder="User Id"
            {...register("User Id", { required: true })}
            className="p-2 rounded-md bg-white border border-gray-300"
          />
        </div>
        <div className="flex flex-col flex-1">
          <label className="mb-1 text-gray-700">Limit</label>
          <input
            type="number"
            placeholder="Limit"
            {...register("Limit", { required: false, min: 1 })}
            className="p-2 rounded-md bg-white border border-gray-300"
          />
        </div>
      </div>
      <div className="flex gap-4">
        <div className="flex flex-col flex-1">
          <label className="mb-1 text-gray-700">Query</label>
          <input
            type="text"
            placeholder="Query"
            {...register("Query", {})}
            className="p-2 rounded-md bg-white border border-gray-300"
          />
        </div>
        <div className="flex flex-col justify-end">
          <input
            type="submit"
            value="Submit"
            className="p-2 rounded-md bg-blue-600 text-white cursor-pointer hover:bg-blue-700"
          />
        </div>
      </div>
    </form>
  );
}


const Memories = () => {

  const columns = [
    {
      name: 'Id',
      selector: (row: any) => row.title,
    },
    {
      name: 'Content',
      selector: (row: any) => row.content,
    },
    {
      name: 'Created At',
      selector: (row: any) => row.createdAt,
    },
    {
      name: 'User Id',
      selector: (row: any) => row.userId,
    },
    {
      name: 'Memory Type',
      selector: (row: any) => row.memoryType,
    },
    {
      name: 'Update At',
      selector: (row: any) => row.updatedAt,
    },
  ];

  const data = [
    {
      id: 1,
      title: 'Memory 1',
      content: 'This is the content of memory 1',
      createdAt: '2023-01-01',
      userId: 'user1',
      memoryType: 'Type A',
      updatedAt: '2023-01-02',
    },
    {
      id: 2,
      title: 'Memory 2',
      content: 'This is the content of memory 2',
      createdAt: '2023-02-01',
      userId: 'user2',
      memoryType: 'Type B',
      updatedAt: '2023-02-02',
    },
    {
      id: 3,
      title: 'Memory 3',
      content: 'This is the content of memory 3',
      createdAt: '2023-03-01',
      userId: 'user3',
      memoryType: 'Type C',
      updatedAt: '2023-03-02',
    },
    {
      id: 4,
      title: 'Memory 4',
      content: 'This is the content of memory 4',
      createdAt: '2023-04-01',
      userId: 'user4',
      memoryType: 'Type D',
      updatedAt: '2023-04-02',
    },
    {
      id: 5,
      title: 'Memory 5',
      content: 'This is the content of memory 5',
      createdAt: '2023-05-01',
      userId: 'user5',
      memoryType: 'Type E',
      updatedAt: '2023-05-02',
    },
  ]

  return (
    <>
      {/* <Breadcrumb pageName="Tables" /> */}
    
      <div className="flex flex-col gap-10">
        <h1 className="text-2xl font-bold text-gray-800">Memories</h1>
        <SearchBar />

        {/* <TableOne />
        <TableTwo />
        <TableThree /> */}
        <DataTable
          columns={columns}
          data={data}
        />
      </div>
    </>
  );
};

export default Memories;
