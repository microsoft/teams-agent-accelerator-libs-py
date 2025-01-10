import { useState } from 'react';
import { FieldValues, SubmitHandler, useForm } from 'react-hook-form';
import Loader from '../components/Loader';
import { searchMemories } from '../api/memoryServer';
import { Memory } from '../types';
import Breadcrumb from '../components/Breadcrumb';
import DataTableBase from '../components/DataTable';
import Dialog from '../components/Dialog'; // Import the Dialog component

interface SearchData {
  userId: string;
  limit: number;
  query: string;
}

interface SearchBarProps {
  onSubmit: SubmitHandler<SearchData>;
}

const SearchBar = ({ onSubmit }: SearchBarProps) => {
  const { register, handleSubmit, formState: { errors } } = useForm();
  
  console.log(errors);

  return (
    <div className="">
      <form onSubmit={handleSubmit(onSubmit as SubmitHandler<FieldValues>)}>
          <div className="flex flex-row gap-5.5">
            <div className="flex flex-col flex-1">
              <label className="mb-1 text-gray-700 font-bold dark:text-white">Query</label>
              <input
                type="text"
                placeholder="What does the user..."
                {...register("query", {})}
                className="p-2 rounded-md bg-white border border-gray-300 rounded-lg border-[1.5px] border-stroke bg-transparent py-3 px-5 text-black outline-none transition focus:border-teams active:border-teams disabled:cursor-default disabled:bg-whiter dark:border-form-strokedark dark:bg-form-input dark:text-white dark:focus:border-teams"
              />
            </div>
            <div className="flex flex-col flex-1">
              <label className="mb-1 text-gray-700 font-bold dark:text-white">User Id</label>
              <input
                type="text"
                placeholder="1234"
                {...register("userId", { required: true })}
                className="p-2 rounded-md bg-white border border-gray-300 rounded-lg border-[1.5px] border-stroke bg-transparent py-3 px-5 text-black outline-none transition focus:border-teams active:border-teams disabled:cursor-default disabled:bg-whiter dark:border-form-strokedark dark:bg-form-input dark:text-white dark:focus:border-teams"
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
                className="p-2 rounded-md bg-white border border-gray-300 rounded-lg border-[1.5px] border-stroke bg-transparent py-3 px-5 text-black outline-none transition focus:border-teams active:border-teams disabled:cursor-default disabled:bg-whiter dark:border-form-strokedark dark:bg-form-input dark:text-white dark:focus:border-teams"
              />
            </div>
            <div className="flex flex-col justify-end">
              <input
                type="submit"
                value="Submit"
                className="inline-flex items-center justify-center rounded-full bg-teams py-3 px-5 text-center font-medium text-white hover:bg-opacity-90 lg:px-8 xl:px-10"
              />
            </div>
          </div>
      </form>
    </div>
  );
}

const dummyData: Memory[] = []

const Memories = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [memories, setMemories] = useState<Memory[]>(dummyData);
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null); // State for selected memory

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

  const handleRowClick = (row: Memory) => {
    setSelectedMemory(row); // Set the selected memory
  };

  const closeDialog = () => {
    setSelectedMemory(null); // Close the dialog
  };

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
      grow: 0.5,
      cell: (row: Memory) => <CustomCell content={row.updated_at ?? "-"} />,
      onClick: handleRowClick // Add onClick handler to rows
    },
  ];

  return (
    <div className="p-10">
      <Breadcrumb pageName="Memories" />
      
      <div className="mb-10"></div>
      
      <div className="mb-6 text-black dark:text-white">
      Search for memories by entering a user ID (required), query, and limit. If no query is provided, all memories for the user will be returned.
      </div>
      
      <div className="flex flex-col gap-6">  
        <SearchBar onSubmit={onSubmit} />
        <DataTableBase
          columns={columns}
          data={memories}
          progressPending={loading}
          progressComponent={<Loader />}
          noDataComponent={<div className="flex items-center justify-center h-50 text-center">No memories found</div>}
          onRowClicked={handleRowClick} // Add onRowClicked handler
        />
        {selectedMemory && (
          <Dialog memory={selectedMemory} onClose={closeDialog} /> // Render Dialog if a memory is selected
        )}
      </div>
    </div>
  );
};

export default Memories;
