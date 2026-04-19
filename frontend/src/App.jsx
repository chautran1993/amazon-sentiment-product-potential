import { useEffect, useMemo, useState } from "react";
import Header from "./components/Header.jsx";
import OverviewDashboard from "./pages/OverviewDashboard.jsx";
import ProductDetail from "./pages/ProductDetail.jsx";
import PredictReview from "./pages/PredictReview.jsx";
import TopProductRanking from "./pages/TopProductRanking.jsx";
import { getProductDetail, getTopRanking } from "./services/api.js";

const tabs = [
  { id: "overview", label: "Overview" },
  { id: "ranking", label: "Top Ranking" },
  { id: "product", label: "Product Detail" },
  { id: "predict", label: "Predict Review" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedAsin, setSelectedAsin] = useState("");
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [productError, setProductError] = useState("");

  useEffect(() => {
    async function loadDefaultProduct() {
      try {
        const products = await getTopRanking(1);
        if (products.length > 0) {
          setSelectedAsin(products[0].asin);
        }
      } catch {
        setSelectedAsin("");
      }
    }
    loadDefaultProduct();
  }, []);

  useEffect(() => {
    async function loadProduct() {
      if (!selectedAsin) {
        setSelectedProduct(null);
        return;
      }
      try {
        setProductError("");
        const product = await getProductDetail(selectedAsin);
        setSelectedProduct(product);
      } catch (error) {
        setSelectedProduct(null);
        setProductError(error.message);
      }
    }
    loadProduct();
  }, [selectedAsin]);

  const currentPage = useMemo(() => {
    if (activeTab === "overview") {
      return <OverviewDashboard />;
    }
    if (activeTab === "ranking") {
      return (
        <TopProductRanking
          onSelectProduct={(asin) => {
            setSelectedAsin(asin);
            setActiveTab("product");
          }}
        />
      );
    }
    if (activeTab === "product") {
      return (
        <ProductDetail
          asin={selectedAsin}
          onAsinChange={setSelectedAsin}
          product={selectedProduct}
          error={productError}
        />
      );
    }
    return <PredictReview />;
  }, [activeTab, productError, selectedAsin, selectedProduct]);

  return (
    <div className="app-shell">
      <Header />
      <nav className="tabs" aria-label="Dashboard sections">
        {tabs.map((tab) => (
          <button
            className={activeTab === tab.id ? "tab active" : "tab"}
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </nav>
      <main>{currentPage}</main>
    </div>
  );
}
