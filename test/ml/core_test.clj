(ns ml.core-test
  (:require [clojure.test :refer :all]
            [clojure.data.csv :as csv]
            [clojure.string :as clj-str]
            [clojure.java.io :as io]
            [ml.core :refer :all]
            [ml.helpers :refer :all]))

(defn traverse-tree [tree opts]
  (cond
    (and (map? tree) (contains? tree :leaf)) (update tree :leaf #((or (:leaf opts) identity) %))
    (and (map? tree) (contains? tree :test)) (into {}
                                                   (map (fn [[tree-key tree-value]]
                                                          (let [keys-in-every-node #{:test :attribute-data :n-data-points}]
                                                            (if (keys-in-every-node tree-key)
                                                              [tree-key ((or (tree-key opts) identity) tree-value)]
                                                              [tree-key (traverse-tree tree-value opts)])))
                                                        tree))
    :else tree))

(defn format-text
  [txt]
  (-> txt
      (clj-str/replace #"\." "-")
      (clj-str/lower-case)))

(defn split-data
  [txt]
  (clj-str/split txt #" "))

(def iris-tree
  {:test "test-fn"
   :attribute-data {:cost 1/3 :data :petal-length :split-point "2.500"}
   :n-data-points 75
   true {:leaf "setosa"}
   false {:test "test-fn"
          :attribute-data {:cost 2/27 :data :petal-length :split-point "4.800"}
          :n-data-points 50
          true {:leaf "versicolor"}
          false {:test "test-fn"
                 :attribute-data {:cost 4/45 :data :petal-width :split-point "1.750"}
                 :n-data-points 27
                 true {:test "test-fn"
                       :attribute-data {:cost 4/15 :data :petal-width :split-point "1.550"}
                       :n-data-points 5
                       true {:leaf "virginica"}
                       false {:leaf "versicolor"}}
                 false {:leaf "virginica"}}}})

(def wrongly-predicted-iris-data-points
  [{:sepal-length (float 6.9) :sepal-width (float 3.1) :petal-length (float 4.9) :petal-width (float 1.5) :species "versicolor"}
   {:sepal-length (float 5.9) :sepal-width (float 3.2) :petal-length (float 4.8) :petal-width (float 1.8) :species "versicolor"}
   {:sepal-length (float 6.3) :sepal-width (float 2.5) :petal-length (float 4.9) :petal-width (float 1.5) :species "versicolor"}
   {:sepal-length (float 6.8) :sepal-width (float 2.8) :petal-length (float 4.8) :petal-width (float 1.4) :species "versicolor"}
   {:sepal-length (float 4.9) :sepal-width (float 2.5) :petal-length (float 4.5) :petal-width (float 1.7) :species "virginica"}])

(deftest test-decision-tree-training
  (let [iris-data (with-open [reader (io/reader "test-resources/iris.csv")]
                    (doall
                      (csv/read-csv reader)))
        parsed-data (apply map
                           (fn [heading & args]
                             (let [first-half (keep identity
                                                    (map-indexed (fn [i v]
                                                                   (when (odd? i)
                                                                     v))
                                                                 args))
                                   second-half (keep identity
                                                     (map-indexed (fn [i v]
                                                                    (when-not (odd? i)
                                                                      v))
                                                                  args))
                                   parse-data #(try (Float. %) (catch Exception e (when-not (empty? %) %)))]
                               {:test {(keyword (format-text heading)) (mapv parse-data first-half)}
                                :train {(keyword (format-text heading)) (mapv parse-data second-half)}}))
                           (map #(split-data (first %)) (butlast iris-data)))
        data-ready-for-training (with-meta (apply merge
                                                  (map #(let [[avain arvo] (first (:test %))]
                                                          {avain (with-meta arvo (if (= avain :species)
                                                                                   {:type :categorical}
                                                                                   {:type :numerical}))})
                                                       parsed-data))
                                           {:y :species})
        tree (train-decision-tree data-ready-for-training {:stop-criterion 3 :cost-function :gini})
        test-data (map-vecs->vec-maps (apply merge (map :train parsed-data)))
        correct-pred (map #(= (apply-tree tree %) (:species %))
                          test-data)]
    (is (= iris-tree (traverse-tree tree {:attribute-data #(update % :split-point (fn [value]
                                                                                    (format "%.3f" value)))
                                          :test (fn [_] "test-fn")})))
    (is wrongly-predicted-iris-data-points (keep #(when (not= (apply-tree tree %) (:species %)) %)
                                                 test-data))))
