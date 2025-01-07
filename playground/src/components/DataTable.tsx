import React from 'react';
import DataTable from 'react-data-table-component';


const DataTableBase = function (props: any) {
	return (
		<DataTable
			pagination
            direction="auto"
            fixedHeaderScrollHeight="300px"
            highlightOnHover
            persistTableHead
            pointerOnHover
            responsive
            striped
            subHeaderAlign="right"
            subHeaderWrap
			{...props}
		/>
	);
}

export default DataTableBase;