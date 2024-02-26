import "./adverticementList.scss";
import Sidebar from "../../components/sidebar/Sidebar";
import Navbar from "../../components/navbar/Navbar";
import Datatable from "../../components/datatable/Datatable";
import AdverticementTable from "../../components/adverticementTable/AdverticementTable";
import Footer from "../../components/footer/Footer";

const AdverticementList = () => {
  return (
    <div className="list">
      <Sidebar />
      <div className="listContainer">
        <Navbar />
        <AdverticementTable />
        <Footer/>
      </div>
    </div>
  );
};

export default AdverticementList;
