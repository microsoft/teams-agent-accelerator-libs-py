const Loader = (props: { fullScreen?: boolean }) => {
    const { fullScreen } = props;

    const surroundingBoxStyle = `flex items-center justify-center bg-white ${fullScreen ? 'h-screen' : 'h-64'}`;

    return (
      <div className={surroundingBoxStyle}>
        <div className="h-16 w-16 animate-spin rounded-full border-4 border-solid border-teams border-t-transparent"></div>
      </div>
    );
  };
  
  export default Loader;
  