(ns ml.tree)

(defn map-vecs->vec-maps [m]
  (let [foo (seq m)
        m-keys (map first foo)
        m-vals (map second foo)]
    (apply map (fn [& args]
                 (zipmap m-keys args))
           m-vals)))

(defn information-gain
  []
  ;;TODO
  )

(declare best-attribute)

(defn gini
  [explained-data [data-key explaining-data]]
  (let [classes (distinct explained-data)
        groups (distinct explaining-data)
        groups-n (reduce (fn [grouping data-point]
                           (update grouping
                                   data-point
                                   (fn [group-count]
                                     (if group-count
                                       (inc group-count) 1))))
                         {} explaining-data)
        foo (map (fn [x y]
                   [x y])
                 explaining-data explained-data)
        bar (group-by identity foo)
        asd (reduce (fn [result [[group-bar class] pairs]]
                      (update result group-bar #(conj % (/ (count pairs) (get groups-n group-bar)))))
                    {} bar)
        gini-index (fn [class-propotions]
                     (apply - 1 (map #(* % %) class-propotions)))]
    (apply + (map (fn [group-bar]
                    (* (/ (get groups-n group-bar)
                          (apply + (vals groups-n)))
                       (gini-index (get asd group-bar))))
                  groups))))

(defmulti best-attribute
          (fn [explained-data explaining-data-sets options]
            (:type (meta explained-data))))

(defmethod best-attribute :categorical
  [explained-data explaining-data-sets {:keys [cost-function] :as options}]
  (let [cost-fn (case cost-function
                  (:gini nil) gini
                  :information-gain information-gain
                  (throw (#?(:clj  Exception.
                             :cljs js/Error.) (str "Can't use " cost-function " as a cost function for categorical explained data. "
                                                   "Valid values are :gini and :information-gain. If left undefined, :gini is used."))))
        costs (map (fn [[data-key explaining-data :as foo-bar]]
                     (let [find-split-point? (= :numerical (-> explaining-data meta :type))
                           split-point-data (when find-split-point?
                                              (let [sorted-values (sort explaining-data)
                                                    mid-values (:values (reduce (fn [{:keys [values previous-value]} value]
                                                                                  {:values (conj values
                                                                                                 (+ previous-value (/ (- value previous-value) 2)))
                                                                                   :previous-value value})
                                                                                {:values []
                                                                                 :previous-value (first sorted-values)}
                                                                                (rest sorted-values)))]
                                                (best-attribute explained-data
                                                                (into {}
                                                                      (map (fn [value]
                                                                             [value (with-meta (map #(> value %) explaining-data)
                                                                                               {:type :categorical})]))
                                                                      mid-values)
                                                                :gini)))]
                       {:cost (cost-fn explained-data [data-key (or (-> split-point-data :data vals first) explaining-data)])
                        :data (into {} [foo-bar])
                        :split-point (-> split-point-data :data keys first)}))
                   explaining-data-sets)
        sorted-costs (sort-by :cost costs)]
    (case cost-function
      (:gini nil) (first sorted-costs)
      :information-gain (last sorted-costs))))

(defmethod best-attribute :numerical
  [explained-data explaining-data-sets options]
  (let [split-value (fn [ys-l ys-r]
                      (let [s-l (apply + ys-l)
                            s-r (apply + ys-r)
                            n-l (count ys-l)
                            n-r (count ys-r)]
                        (+ (/ (* s-l s-l)
                              n-l)
                           (/ (* s-r s-r)
                              n-r))))
        possible-split-points (map (fn [[data-key explaining-data :as foo-bar]]
                                     (case (-> foo-bar meta :type)
                                       :numerical (let [foo (map-vecs->vec-maps {:x explaining-data
                                                                                 :y explained-data})
                                                        bar (sort-by :x foo)]
                                                    (reduce (fn [{:keys [i best-split-value previous-split-point] :as split}
                                                                 {split-point :x y :y}]
                                                              (let [ys-l (take i (map :y bar))
                                                                    ys-r (drop i (map :y bar))
                                                                    new-split-value (split-value ys-l ys-r)]
                                                                (if (> new-split-value best-split-value)
                                                                  {:i (inc i)
                                                                   :best-split-value new-split-value
                                                                   :split-point (/ (+ split-point previous-split-point)
                                                                                   2)
                                                                   :previous-split-point split-point}
                                                                  (assoc split
                                                                    :i (inc i)
                                                                    :previous-split-point split-point))))
                                                            {:i 0 :best-split-value 0
                                                             :previous-split-point 0} bar))
                                       :categorical (let [foo (reduce (fn [{i :is :as result} x]
                                                                        (let [y (get explained-data i)]
                                                                          (update {} x #(if (nil? %) y (+ % y)))))
                                                                      {:i 0} explaining-data)
                                                          categories-in-order (map val
                                                                                   (sort-by val (dissoc foo :i)))]
                                                      )
                                       ))
                                   explaining-data-sets)]
    (last (sort-by :best-split-value possible-split-points))))

(defn most-common-value
  [values]
  (key (apply max-key #(-> % val count) (group-by identity values))))

(defn attribute-branches
  [[data-key explaining-data] {binary-tree? :binary-tree?}]
  (case (:type (meta explaining-data))
    (:numerical binary-tree?) [true false]
    :categorical (distinct explaining-data)))

(defn attribute-test
  [{:keys [split-point data]}]
  (fn [data-point]
    (if split-point
      (->> data keys first (get data-point) (> split-point))
      (->> data keys first (get data-point)))))

(defn form-branch-data
  [data explained-variable test branch]
  (let [values-to-keep (map #(= branch (test %)) (map-vecs->vec-maps data))]
    (with-meta
      (reduce-kv (fn [m k v]
                   (assoc m k (with-meta (vec (keep identity
                                                    (map #(when %2 %1) v values-to-keep)))
                                         (meta v))))
                 {} data)
      (meta data))))

(defn train-decision-tree
  [data {:keys [stop-criterion] :as options}]
  ;; Data on muotoa ^{:y :jotain}{:data-sarakkeen-nimi ^{:type :categorical}["punainen" "keltainen"]}
  (let [explained-variable (:y (meta data))
        explained-data (explained-variable data)
        explaining-data-sets (dissoc data explained-variable)]
    (if (or (<= (count explained-data) stop-criterion)
            (= 1 (count (into #{} explained-data))))
      {:leaf (most-common-value explained-data)}
      (let [attribute-data (best-attribute explained-data explaining-data-sets options)
            _ (println "ATTRIBUTE DATA")
            _ (clojure.pprint/pprint attribute-data)
            branches (attribute-branches (first (:data attribute-data)) options)
            test (attribute-test attribute-data)
            node (apply merge
                        {:test test
                         :attribute-data (update attribute-data :data #(-> % keys first))
                         :n-data-points (or (:n-new-data-points options) (-> data vals first count))}
                        (map (fn [branch]
                               (let [new-data (form-branch-data data explained-variable test branch)
                                     n-new-data-points (-> new-data vals first count)
                                     n-data-points (-> data vals first count)]
                                 (cond
                                   (= 0 n-new-data-points) {:leaf nil
                                                            :n-data-points n-new-data-points}
                                   (= n-data-points n-new-data-points) {:leaf (first explained-data)
                                                                        :n-data-points n-new-data-points}
                                   :else {branch (train-decision-tree (form-branch-data data explained-variable test branch) (assoc options
                                                                                                                               :n-data-points n-new-data-points))})))
                             branches))]
        (if (and (contains? node :test)
                 (contains? node :leaf))
          (select-keys node [:leaf])
          node)))))

(defn apply-tree [tree data]
  (if-let [test (:test tree)]
    (apply-tree (get tree (test data)) data)
    (:leaf tree)))