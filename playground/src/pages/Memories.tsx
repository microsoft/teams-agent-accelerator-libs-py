// import Breadcrumb from '../components/Breadcrumbs/Breadcrumb';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import Loader from '../components/Loader';
import { searchMemories } from '../api/memoryServer';
import { Memory } from '../types';
import Breadcrumb from '../components/Breadcrumb';
import DataTableBase from '../components/DataTable';

interface SearchData {
  userId: string;
  limit: number;
  query: string;
}

interface SearchBarProps {
  onSubmit: (data: SearchData | any) => void;
}

const SearchBar = ({ onSubmit }: SearchBarProps) => {
  const { register, handleSubmit, formState: { errors } } = useForm();
  
  console.log(errors);

  /*
<div className="rounded-sm border border-stroke bg-white shadow-default dark:border-strokedark dark:bg-boxdark">
            <div className="border-b border-stroke py-4 px-6.5 dark:border-strokedark">
              <h3 className="font-medium text-black dark:text-white">
                Input Fields
              </h3>
            </div>
            <div className="flex flex-col gap-5.5 p-6.5">
              <div>
                <label className="mb-3 block text-black dark:text-white">
                  Default Input
                </label>
                <input
                  type="text"
                  placeholder="Default Input"
                  className="w-full rounded-lg border-[1.5px] border-stroke bg-transparent py-3 px-5 text-black outline-none transition focus:border-primary active:border-primary disabled:cursor-default disabled:bg-whiter dark:border-form-strokedark dark:bg-form-input dark:text-white dark:focus:border-primary"
                />
              </div>

              <div>
                <label className="mb-3 block text-black dark:text-white">
                  Active Input
                </label>
                <input
                  type="text"
                  placeholder="Active Input"
                  className="w-full rounded-lg border-[1.5px] border-primary bg-transparent py-3 px-5 text-black outline-none transition focus:border-primary active:border-primary disabled:cursor-default disabled:bg-whiter dark:bg-form-input dark:text-white"
                />
              </div>

              <div>
                <label className="mb-3 block font-medium text-black dark:text-white">
                  Disabled label
                </label>
                <input
                  type="text"
                  placeholder="Disabled label"
                  disabled
                  className="w-full rounded-lg border-[1.5px] border-stroke bg-transparent py-3 px-5 text-black outline-none transition focus:border-primary active:border-primary disabled:cursor-default disabled:bg-whiter dark:border-form-strokedark dark:bg-form-input dark:text-white dark:focus:border-primary dark:disabled:bg-black"
                />
              </div>
            </div>
          </div>
  */

  return (
    <div className="">
      <form onSubmit={handleSubmit(onSubmit)}>
          <div className="flex flex-row gap-5.5">
            <div className="flex flex-col flex-1">
              <label className="mb-1 text-gray-700 font-bold dark:text-white">Query</label>
              <input
                type="text"
                placeholder="What does the user..."
                {...register("query", {})}
                className="p-2 rounded-md bg-white border border-gray-300 rounded-lg border-[1.5px] border-stroke bg-transparent py-3 px-5 text-black outline-none transition focus:border-primary active:border-primary disabled:cursor-default disabled:bg-whiter dark:border-form-strokedark dark:bg-form-input dark:text-white dark:focus:border-primary"
              />
            </div>
            <div className="flex flex-col flex-1">
              <label className="mb-1 text-gray-700 font-bold dark:text-white">User Id</label>
              <input
                type="text"
                placeholder="1234"
                {...register("userId", { required: true })}
                className="p-2 rounded-md bg-white border border-gray-300 rounded-lg border-[1.5px] border-stroke bg-transparent py-3 px-5 text-black outline-none transition focus:border-primary active:border-primary disabled:cursor-default disabled:bg-whiter dark:border-form-strokedark dark:bg-form-input dark:text-white dark:focus:border-primary"
              />
            </div>
            <div className="flex flex-col flex-0.5">
              <label className="mb-1 text-gray-700 font-bold dark:text-white">Limit</label>
              <input
                type="number"
                placeholder="Limit"
                min={1}
                defaultValue={5}
                {...register("limit", { required: false, min: 1 })}
                className="p-2 rounded-md bg-white border border-gray-300 rounded-lg border-[1.5px] border-stroke bg-transparent py-3 px-5 text-black outline-none transition focus:border-primary active:border-primary disabled:cursor-default disabled:bg-whiter dark:border-form-strokedark dark:bg-form-input dark:text-white dark:focus:border-primary"
              />
            </div>
            <div className="flex flex-col justify-end">
              <input
                type="submit"
                value="Submit"
                className="inline-flex items-center justify-center rounded-full bg-meta-3 py-3 px-5 text-center font-medium text-white hover:bg-opacity-90 lg:px-8 xl:px-10"
              />
            </div>
          </div>
      </form>
    </div>
  );
}

const dummyData: Memory[] = []
// const dummyData: Memory[] = [
//   {
//     id: "1",
//     content: 'This is the content of memory 1',
//     created_at: '2023-01-01',
//     user_id: 'user1',
//     memory_type: 'semantic',
//     updated_at: '2023-01-02',
//     message_attributions: [],
//   },
//   {
//     id: "2",
//     content: 'This is the content of memory 2',
//     created_at: '2023-02-01',
//     user_id: 'user2',
//     memory_type: 'semantic',
//     updated_at: '2023-02-02',
//     message_attributions: [],
//   }
// ]


const Memories = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [memories, setMemories] = useState<Memory[]>(dummyData);

  const onSubmit = async (data: SearchData) => {
    console.log(data);
    
    // Simulate loading
    setLoading(true);
    setTimeout(() => setLoading(false), 1000);

    // Call API to update data
    const result = await searchMemories(data.query, data.userId, data.limit);

    // Update data
    setMemories(result);
  }

  const ColumnTitle = (props: { title: string }) => {
    const { title } = props;

    return (
      <div className="mb-1 text-gray-700 font-bold">
        {title}
      </div>
    );
  }

  const CustomCell = (props: { content: string }) => {
    const { content } = props;

    return (
      <div className="text-center">
        { content }
      </div>
    );
  }

  const columns = [
    {
      name: <ColumnTitle title="ID" />,
      selector: (row: Memory) => row.id,
      grow: 0.5,
      cell: (row: Memory) => <CustomCell content={row.id} />
    },
    {
      name: <ColumnTitle title="Content" />,
      selector: (row: Memory) => row.content,
      grow: 1,
    },
    {
      name: <ColumnTitle title="Created At" />,
      selector: (row: Memory) => row.created_at,
      grow: 0.5,
    },
    {
      name: <ColumnTitle title="User Id" />,
      selector: (row: Memory) => row.user_id,
      grow: 0.5
    },
    {
      name: <ColumnTitle title="Memory Type" />,
      selector: (row: Memory) => row.memory_type,
      grow: 0.5
    },
    {
      name: <ColumnTitle title="Update At" />,
      selector: (row: Memory) => row.updated_at ?? "-",
      grow: 0.5
    },
  ];

  return (
    <div className="p-10">
      <Breadcrumb pageName="Memories" />
      <div className="mb-10"></div>
      
      <div className="flex flex-col gap-10">  
        <SearchBar onSubmit={onSubmit} />
        <DataTableBase
          columns={columns}
          data={memories}
          progressPending={loading}
          progressComponent={<Loader />}
          noDataComponent={<div className="flex items-center justify-center h-50 text-center">No memories found</div>}
        />
      </div>
    </div>
  );
};

export default Memories;
