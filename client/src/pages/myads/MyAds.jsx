import "./myads.scss";
import Sidebar from "../../components/sidebar/Sidebar";
import Navbar from "../../components/navbar/Navbar";
import Datatable from "../../components/datatable/Datatable";
import MyAdsTable from "../../components/myadsTable/MyAdsTable";
import Footer from "../../components/footer/Footer";


const MyAds = () => {
  return (
    <div className="list">
      <Sidebar />
      <div className="listContainer">
        <Navbar />
        < MyAdsTable/>
        <Footer/>
      </div>
    </div>
  );
};

export default MyAds;
