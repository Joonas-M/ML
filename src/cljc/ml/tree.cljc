(ns ml.tree)

(defn map-vecs->vec-maps [m]
  (let [foo (seq m)
        m-keys (map first foo)
        m-vals (map second foo)]
    (apply map (fn [& args]
                 (zipmap m-keys args))
           m-vals)))

(defn traverse-tree [tree {:keys [recur-fn] :as opts}]
  (cond
    (and (map? tree) (contains? tree :leaf)) (update tree :leaf #((or (:leaf opts) identity) %))
    (and (map? tree) (contains? tree :test)) (into {}
                                                   (map (fn [[tree-key tree-value]]
                                                          (let [keys-in-every-node #{:test :attribute-data :n-data-points}]
                                                            (if (keys-in-every-node tree-key)
                                                              [tree-key ((or (tree-key opts) identity) tree-value)]
                                                              [tree-key ((or recur-fn traverse-tree) tree-value opts)])))
                                                        tree))
    :else tree))

(defn test-tree [test-fn tree {:keys [result] :as opts}]
  (let [possible-node (if (= result false)
                        false
                        (traverse-tree tree (assoc opts :recur-fn (partial test-tree test-fn)
                                                        :result (test-fn tree))))]
    (if (boolean? possible-node)
      possible-node
      ;; If result is nil, then we are dealing with the root node
      (if (nil? result)
        (test-fn possible-node)
        result))))

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

(defn split-categorical-data
  "Splits a collection of categorical values into two sets and return the first set"
  [explained-data explaining-data]
  (let [foo (reduce (fn [{i :i :as result} x]
                      (let [y (get explained-data i)]
                        (-> result
                            (update x #(if (nil? %)
                                         {:s y :n 1}
                                         {:s (+ (:s %) y) :n (inc (:n %))}))
                            (update i inc))))
                    {:i 0}
                    explaining-data)
        categories-in-order (sort-by #(-> % val :s) (dissoc foo :i))]
    (reduce (fn [{:keys [previous-s-l previous-s-r previous-n-l previous-n-r
                         best-split-value previous-categories] :as previous-values}
                 [k {:keys [s n]}]]
              (let [s-l (+ previous-s-l s)
                    s-r (- previous-s-r s)
                    n-l (+ previous-n-l n)
                    n-r (- previous-n-r n)
                    new-split-value (+ (/ (* s-l s-l)
                                          n-l)
                                       (/ (* s-r s-r)
                                          n-r))]
                (if (> new-split-value best-split-value)
                  {:best-split-value new-split-value
                   :split-point (conj previous-categories k)
                   :previous-s-l s-l :previous-s-r s-r :previous-n-l n-l
                   :previous-n-r n-r :previous-categories (conj previous-categories k)}
                  (merge previous-values
                         {:previous-s-l s-l :previous-s-r s-r :previous-n-l n-l
                          :previous-n-r n-r :previous-categories (conj previous-categories k)}))))
            {:previous-s-l 0 :previous-s-r (apply + (map #(-> % val :s) categories-in-order))
             :previous-n-l 0 :previous-n-r (apply + (map #(-> % val :n) categories-in-order))
             :best-split-value 0 :previous-categories #{}}
            categories-in-order)))

(defmulti best-attribute
          (fn [explained-data explaining-data-sets options]
            (:type (meta explained-data))))

(defmethod best-attribute :categorical
  [explained-data explaining-data-sets {:keys [cost-function binary-tree? find-split-point?] :as options}]
  (let [cost-fn (case cost-function
                  (:gini nil) gini
                  :information-gain information-gain
                  (throw (#?(:clj  Exception.
                             :cljs js/Error.) (str "Can't use " cost-function " as a cost function for categorical explained data. "
                                                   "Valid values are :gini and :information-gain. If left undefined, :gini is used."))))
        possible-split-points (map (fn [[data-key explaining-data :as explaining-data-vec]]
                                     (let [explaining-data-type (-> explaining-data meta :type)
                                           split-point-data (case explaining-data-type
                                                              :numerical (let [sorted-values (sort explaining-data)
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
                                                                                           {:cost-function :gini
                                                                                            :find-split-point? true}))
                                                              :categorical (when binary-tree?
                                                                             (let [split-point (:split-point (split-categorical-data explained-data explaining-data))]
                                                                               {:explaining-data-map {data-key (map split-point explaining-data)}})))]
                                       (merge {:cost (cost-fn explained-data [data-key (or (-> split-point-data :explaining-data-map vals first)
                                                                                           explaining-data)])
                                               :split-point (-> split-point-data :explaining-data-map keys first)
                                               :branches (if (and (= :categorical explaining-data-type)
                                                                  (not= binary-tree?))
                                                           (distinct explaining-data)
                                                           [true false])
                                               :data-key data-key}
                                              (when find-split-point?
                                                {:explaining-data-map (into {} [explaining-data-vec])}))))
                                   explaining-data-sets)
        sorted-possible-split-points (sort-by :cost possible-split-points)]
    (case cost-function
      (:gini nil) (first sorted-possible-split-points)
      :information-gain (last sorted-possible-split-points))))

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
        possible-split-points (map (fn [[data-key explaining-data :as explaining-data-vec]]
                                     (let [explaining-data-type (-> explaining-data meta :type)
                                           split-point-data (case (-> explaining-data meta :type)
                                                              :numerical (let [data-sorted-by-explaining-data (sort-by :x (map-vecs->vec-maps {:x explaining-data
                                                                                                                                               :y explained-data}))]
                                                                           (reduce (fn [{:keys [i best-split-value previous-split-point] :as split}
                                                                                        {split-point :x}]
                                                                                     (let [ys-l (take i (map :y data-sorted-by-explaining-data))
                                                                                           ys-r (drop i (map :y data-sorted-by-explaining-data))
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
                                                                                    :previous-split-point 0}
                                                                                   data-sorted-by-explaining-data))
                                                              :categorical (split-categorical-data explained-data explaining-data))]
                                       {:best-split-value (:best-split-value split-point-data)
                                        :split-point (:split-point split-point-data)
                                        :branches [true false]
                                        :data-key data-key}))
                                   explaining-data-sets)]
    (last (sort-by :best-split-value possible-split-points))))

(defn most-common-value
  [values]
  (key (apply max-key #(-> % val count) (group-by identity values))))

(defn attribute-test
  [{:keys [split-point data-key]}]
  (let [test-fn (cond
                  (number? split-point) #(> split-point %)
                  (set? split-point) #(split-point %)
                  :else identity)]
    (fn [data-point]
      (test-fn (get data-point data-key)))))

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
            branches (:branches attribute-data)
            test (attribute-test attribute-data)
            node (apply merge
                        {:test test
                         :attribute-data attribute-data
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