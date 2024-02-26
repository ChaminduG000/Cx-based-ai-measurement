import "./widget.scss";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import PersonOutlinedIcon from "@mui/icons-material/PersonOutlined";
import AccountBalanceWalletOutlinedIcon from "@mui/icons-material/AccountBalanceWalletOutlined";
import ShoppingCartOutlinedIcon from "@mui/icons-material/ShoppingCartOutlined";
import MonetizationOnOutlinedIcon from "@mui/icons-material/MonetizationOnOutlined";
import SentimentVerySatisfiedIcon from "@mui/icons-material/SentimentVerySatisfied";
import SentimentVeryDissatisfiedIcon from "@mui/icons-material/SentimentVeryDissatisfied";
import SentimentNeutralIcon from "@mui/icons-material/SentimentNeutral";
import GroupIcon from "@mui/icons-material/Group";
import ScheduleSendIcon from "@mui/icons-material/ScheduleSend";
import FeaturedVideoIcon from "@mui/icons-material/FeaturedVideo";
import AddReactionIcon from '@mui/icons-material/AddReaction';
import { useEffect, useState } from "react";
import axios from "axios";
const Widget = ({ type }) => {
  const [user, setUser] = useState([]);
  const [adverticement, setAdverticement] = useState([]);
  const [useradverticement, setUserAdverticement] = useState([]);
  let apiUrl = "http://localhost:8800/api/users";

  console.log(apiUrl);
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(apiUrl);
        setUser(response.data);
      } catch (error) {
        console.error("Error fetching data", error);
      }
    };

    fetchData();
  }, [apiUrl]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(
          "http://localhost:8800/api/adverticements/"
        );
        setAdverticement(response.data);
      } catch (error) {
        console.error("Error fetching data", error);
      }
    };

    fetchData();
  }, [apiUrl]);
  const currentUser = JSON.parse(localStorage.getItem("currentUser"));
  useEffect(() => {
    const fetchDUserAdsData = async () => {
      try {
        const response = await axios.get(
          `http://localhost:8800/api/adverticements/user/${currentUser._id}`
        );
        setUserAdverticement(response.data);
      } catch (error) {
        console.error("Error fetching data", error);
      }
    };

    fetchDUserAdsData();
  }, [apiUrl]);

  const filteredApprovedAdverticement = useradverticement.filter(
    (item) => item.status === true
  );
  const filteredPendingAdverticement = useradverticement.filter(
    (item) => item.status === false
  );
    const filteredPendingAllAdverticement = adverticement.filter(
    (item) => item.status === false
  );
  const filteredData =
    currentUser.role === "admin"
      ? adverticement.filter((item) => item.status === true)
      : adverticement;
  let data;

  //temporary
  const amount = 100;
  const diff = 20;

  switch (type) {
    case "user":
      data = {
        title: "CUSTOMERS",
        isMoney: false,
        counter: user.length,
        // link: "See all users",
        icon: (
          <GroupIcon
            className="icon"
            style={{
              color: "crimson",
              backgroundColor: "#f6e58d",
            }}
          />
        ),
      };
      break;
    case "order":
      data = {
        title: "ADVERTISMENTS",
        isMoney: false,
        counter: filteredData.length,
        // link: "View all adverticements",
        icon: (
          <FeaturedVideoIcon
            className="icon"
            style={{
              backgroundColor: "#4834d4",
              color: "white",
            }}
          />
        ),
      };
      break;
      case "history":
        data = {
          // title: "REACTIONS",
          // isMoney:false,
           icon: (
          <AddReactionIcon
            // className="icon"
            style={{
              // backgroundColor: "hsl(128, 98%, 84%)",
              // color: "green",
            }}
          />
        ),
        };

      break;
    case "balance":
      data = {
        title: "REQUESTS",
        isMoney: false,
        // link: "See requests",
        counter: filteredPendingAllAdverticement.length,
        icon: (
          <ScheduleSendIcon
            className="icon"
            style={{
              backgroundColor: "rgba(128, 0, 128, 0.2)",
              color: "purple",
            }}
          />
        ),
      };
      break;
    case "myadsApprove":
      data = {
        title: "MY ACTIVATED ADVERTISMENTS",
        isMoney: false,
        // link: "See requests",
        counter: filteredApprovedAdverticement.length,
        icon: (
          <ScheduleSendIcon
            className="icon"
            style={{
              backgroundColor: "rgba(128, 0, 128, 0.2)",
              color: "purple",
            }}
          />
        ),
      };
      break;
    case "myadsInactive":
      data = {
        title: "MY INACTIVATED ADVERTISMENTS",
        isMoney: false,
        // link: "See requests",
        counter: filteredPendingAdverticement.length,
        icon: (
          <ScheduleSendIcon
            className="icon"
            style={{
              backgroundColor: "rgba(128, 0, 128, 0.2)",
              color: "purple",
            }}
          />
        ),
      };
      break;
    default:
      break;
  }

  return (
    <div className="widget">
      <div className="left">
        <span className="title">{data.title}</span>
        <span className="counter">{data.counter}</span>
        <span className="link">{data.link}</span>
      </div>
      <div className="right">
        <div className="percentage positive">
          <KeyboardArrowUpIcon />
          {diff} %
        </div>
        {data.icon}
      </div>
    </div>
  );
};

export default Widget;
