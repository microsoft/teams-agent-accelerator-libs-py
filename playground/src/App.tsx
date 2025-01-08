import { useEffect, useState } from 'react';
import { Route, Routes, useLocation, Navigate } from 'react-router-dom';

import Loader from './components/Loader';
import PageTitle from './components/PageTitle';
import DefaultLayout from './layout/DefaultLayout';
import Memories from './pages/Memories';

function App() {
  const [loading, setLoading] = useState<boolean>(true);
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  useEffect(() => {
    setTimeout(() => setLoading(false), 1000);
  }, []);

  return loading ? (
    <Loader fullScreen={true} />
  ) : (
    <DefaultLayout>
      <Routes>
        <Route path="/" element={<Navigate to="/memories" />} />
        <Route
          path="/memories"
          element={
            <>
              <PageTitle title="Memories Playground" />
              <Memories />
            </>
          }
        />
      </Routes>
    </DefaultLayout>
  );
}

export default App;
